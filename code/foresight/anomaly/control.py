"""
Control charts on daily counts: an upper limit for each series and
each day, computed from the days before it, and an alert when a day's
count goes over.

Two ways to set the limit from the trailing baseline:

    "sigma"    mean + k standard deviations, the textbook chart
    "poisson"  the count a Poisson series with that mean exceeds with
               probability at most `alpha`

The sigma chart assumes counts spread like a bell curve. A supplier
with half a ticket a day does not: its standard deviation is often
zero over a quiet month, and then a single ticket crosses the limit.
The Poisson chart reads the spread off the mean, as counts of rare
events do.

With `skip_alerts=True` a day that raised an alert is left out of the
baseline of the days after it, so a long spike cannot become its own
normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

THREE_SIGMA = 0.00135       # one-sided tail beyond 3 sd of a bell curve


@dataclass
class Chart:
    """What a control chart saw: counts, centre line, limit, alerts."""
    counts: pd.DataFrame
    centre: pd.DataFrame
    upper: pd.DataFrame
    alerts: pd.DataFrame


def poisson_limit(mean, alpha: float, window: int):
    """The count a Poisson series exceeds with probability <= alpha.
    A mean of zero is read as one ticket in the window: a series that
    has been silent is quiet, not impossible."""
    lam = np.maximum(np.asarray(mean, dtype=float), 1.0 / window)
    return poisson.isf(alpha, lam)


def _limit(values, method, k, alpha, window):
    mean = values.mean()
    if method == "sigma":
        return mean, mean + k * values.std(ddof=1)
    return mean, float(poisson_limit(mean, alpha, window))


def control_chart(counts: pd.DataFrame, window: int = 56,
                  method: str = "poisson", k: float = 3.0,
                  alpha: float = THREE_SIGMA,
                  skip_alerts: bool = False) -> Chart:
    """Limits for every series from the `window` days before each day.
    Days without a full baseline get no limit and never alert."""
    if method not in ("sigma", "poisson"):
        raise ValueError(f"unknown method {method!r}")
    if not skip_alerts:
        past = counts.shift(1).rolling(window, min_periods=window)
        centre = past.mean()
        if method == "sigma":
            upper = centre + k * past.std(ddof=1)
        else:
            upper = pd.DataFrame(poisson_limit(centre, alpha, window),
                                 index=counts.index,
                                 columns=counts.columns)
            upper = upper.where(centre.notna())
        alerts = (counts > upper) & upper.notna()
        return Chart(counts, centre, upper, alerts)

    centre = pd.DataFrame(np.nan, index=counts.index,
                          columns=counts.columns)
    upper = centre.copy()
    for col in counts.columns:
        x = counts[col].to_numpy()
        kept: list[int] = []            # baseline: non-alert days only
        c_col = np.full(len(x), np.nan)
        u_col = np.full(len(x), np.nan)
        for i, v in enumerate(x):
            if len(kept) >= window:
                base = np.asarray(kept[-window:], dtype=float)
                c_col[i], u_col[i] = _limit(base, method, k, alpha,
                                            window)
                if v > u_col[i]:
                    continue            # an alert day stays out
            kept.append(v)
        centre[col], upper[col] = c_col, u_col
    alerts = (counts > upper) & upper.notna()
    return Chart(counts, centre, upper, alerts)
