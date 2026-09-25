"""
Prediction intervals: how often they held, and how to widen them from
their own record.

The backtest's quantile columns, q10 and q90, bound an 80% interval.
coverage() counts how often the actual landed inside. widen() stretches
each interval by the amount that would have made the intervals before
it cover 80%, measured only on targets already on record at its origin,
horizon by horizon. That second step is conformal calibration in its
simplest form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pinball(actual, forecast, alpha: float) -> float:
    """Quantile loss: an actual above the forecast costs alpha per
    unit, one below it 1 - alpha. Its minimum is the alpha quantile."""
    e = np.asarray(actual, float) - np.asarray(forecast, float)
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def coverage(r: pd.DataFrame, lo: str = "q10",
             hi: str = "q90") -> float:
    """The share of actuals inside [lo, hi]."""
    return float(((r.actual >= r[lo]) & (r.actual <= r[hi])).mean())


def widen(r: pd.DataFrame, target: float = 0.8,
          least: int = 200) -> pd.DataFrame:
    """Adds lo and hi: q10 and q90 moved apart (or together) by a
    multiple of the forecast, chosen from earlier misses. Rows with
    fewer than `least` earlier misses to learn from get NaN."""
    r = r.copy()
    # How far outside its interval each actual fell, as a fraction of
    # the forecast; negative when it landed inside.
    r["outside"] = np.maximum(r.q10 - r.actual,
                              r.actual - r.q90) / r.forecast
    r["lo"] = r["hi"] = np.nan
    for (h, origin), rows in r.groupby(["h", "origin"]).groups.items():
        known = r.outside[(r.h == h) & (r.target <= origin)]
        if len(known) < least:
            continue
        stretch = np.quantile(known, target)
        r.loc[rows, "lo"] = r.q10[rows] - stretch * r.forecast[rows]
        r.loc[rows, "hi"] = r.q90[rows] + stretch * r.forecast[rows]
    return r.drop(columns="outside")
