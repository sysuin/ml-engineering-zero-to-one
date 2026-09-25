"""
The expanding-window backtest, and the scores it reports.

For every target month in 2024 and 2025 and every horizon h, stand at
the origin h months before the target, hand the forecaster the history
up to and including the origin, and record what it said beside what
happened. The window expands: each origin sees everything before it.
Every horizon is scored on the same 24 target months.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TARGETS = pd.period_range("2024-01", "2025-12", freq="M")
HORIZONS = (1, 2, 3)


def backtest(panel: pd.DataFrame, forecaster: Callable,
             horizons=HORIZONS, targets=TARGETS) -> pd.DataFrame:
    """One row per series, target and horizon: actual and forecast."""
    rows = []
    for h in horizons:
        for target in targets:
            origin = target - h
            history = panel.loc[:origin]
            forecast = forecaster(history, h)
            rows.append(pd.DataFrame({
                "series": list(panel.columns), "h": h,
                "origin": origin, "target": target,
                "actual": panel.loc[target].to_numpy(float),
                "forecast": forecast.reindex(panel.columns)
                                    .to_numpy(float)}))
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["actual"])


def mae(actual, forecast) -> float:
    """Mean absolute error: the average miss, in units."""
    a, f = _arrays(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def mape(actual, forecast) -> float:
    """Mean absolute percentage error, as a fraction. Undefined where
    anything actual is zero, and dominated by the smallest actuals."""
    a, f = _arrays(actual, forecast)
    return float(np.mean(np.abs(a - f) / np.abs(a)))


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: every miss, in units, over
    every unit sold. A series counts in proportion to its volume."""
    a, f = _arrays(actual, forecast)
    return float(np.sum(np.abs(a - f)) / np.sum(np.abs(a)))


def _arrays(actual, forecast):
    """Plain arrays, refusing a missing forecast rather than skipping
    it: a score that ignores the months a method could not forecast
    flatters the method."""
    a = np.asarray(actual, float)
    f = np.asarray(forecast, float)
    if np.isnan(f).any():
        raise ValueError("a forecast is missing; score common rows")
    return a, f


def by_horizon(results: dict, metric: Callable = wape) -> pd.DataFrame:
    """One row per forecaster, one column per horizon."""
    table = {name: r.groupby("h").apply(
                 lambda g: metric(g.actual, g.forecast))
             for name, r in results.items()}
    return pd.DataFrame(table).T.rename(columns=lambda h: f"h={h}")
