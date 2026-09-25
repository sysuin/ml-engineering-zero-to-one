"""
The demand forecast's error as the actuals arrive. Chapter 24.

Chapter 17's backtest is already what production would have done: at
the end of every month the global model is refitted on everything up to
it and forecasts the month after. So the forecast Meridian received in
2025 is the 2025 part of that record, and monitoring it means reading
the record one month at a time, as each month's sales come in.

Two things are watched, one month ahead, on the established products:

    error      the month's WAPE, against the twelve months before
               it: an alert above their median plus three robust
               standard deviations (1.4826 times the median absolute
               deviation, which one bad month cannot inflate)
    coverage   the share of actuals inside the 80% intervals over the
               last six months: an alert below 60%

widen_recent() is Chapter 17's widen() with a memory: it learns the
stretch only from the misses of the last `months` origins, so a year of
shocks is forgotten a year later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import seasonal_naive
from foresight.forecast.intervals import coverage, widen
from foresight.forecast.model import backtest_model
from foresight.forecast.series import established

COVERAGE_FLOOR = 0.60
RECENT = 6


def record(panel: pd.DataFrame, extra: tuple) -> pd.DataFrame:
    """Chapter 17's record, one month ahead, for the established
    series of `panel`: the shipped model's median and quantiles (fitted
    for all three horizons, as shipped), both widenings, and seasonal
    naive beside every row."""
    r = backtest_model(panel, extra, alphas=(None, 0.1, 0.9))
    kept = established(panel).columns
    r = widen_recent(widen(r[r.series.isin(kept)]))
    r = r[r.h == 1]
    naive = backtest(panel[kept], seasonal_naive, horizons=(1,))
    return r.merge(naive[["series", "target", "forecast"]]
                   .rename(columns={"forecast": "naive"}),
                   on=["series", "target"])


def by_month(r: pd.DataFrame) -> pd.DataFrame:
    """One row per target month: WAPE, seasonal naive's WAPE, and the
    coverage of each interval the record carries."""
    out = []
    for target, g in r.groupby("target"):
        row = {"target": target, "wape": wape(g.actual, g.forecast),
               "naive": wape(g.actual, g.naive),
               "bias": g.forecast.sum() / g.actual.sum() - 1}
        for lo, hi, name in (("q10", "q90", "raw"),
                             ("lo", "hi", "widened"),
                             ("lo_recent", "hi_recent", "recent")):
            if lo in g and g[lo].notna().all():
                row[name] = coverage(g, lo, hi)
        out.append(row)
    return pd.DataFrame(out).set_index("target")


def watch(months: pd.DataFrame, live_from) -> pd.DataFrame:
    """Alerts for each month from `live_from`: WAPE above its limit
    from the twelve months before, or coverage of the widened
    intervals over the last RECENT months below COVERAGE_FLOOR."""
    live = pd.Period(live_from, freq="M")
    m = months[months.index >= live].copy()
    limit = []
    for target in m.index:
        past = months.wape[(months.index < target)
                           & (months.index >= target - 12)]
        mad = (past - past.median()).abs().median()
        limit.append(past.median() + 3 * 1.4826 * mad)
    m["limit"] = limit
    m["error_alert"] = m.wape > m.limit
    rolled = months.widened.rolling(RECENT).mean()
    m["coverage_6m"] = rolled.reindex(m.index)
    m["coverage_alert"] = m.coverage_6m < COVERAGE_FLOOR
    return m


def widen_recent(r: pd.DataFrame, months: int = 12,
                 target: float = 0.8, least: int = 200
                 ) -> pd.DataFrame:
    """Adds lo_recent and hi_recent: q10 and q90 stretched by the
    80th percentile of how far outside its interval each actual fell,
    over the targets of the last `months` months known at the origin."""
    r = r.copy()
    r["outside"] = np.maximum(r.q10 - r.actual,
                              r.actual - r.q90) / r.forecast
    r["lo_recent"] = r["hi_recent"] = np.nan
    for (h, origin), rows in r.groupby(["h", "origin"]).groups.items():
        seen = ((r.h == h) & (r.target <= origin)
                & (r.target > origin - months))
        known = r.outside[seen]
        if len(known) < least:
            continue
        stretch = np.quantile(known, target)
        r.loc[rows, "lo_recent"] = (r.q10[rows]
                                    - stretch * r.forecast[rows])
        r.loc[rows, "hi_recent"] = (r.q90[rows]
                                    + stretch * r.forecast[rows])
    return r.drop(columns="outside")
