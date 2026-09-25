"""
Forecasts that need no model, and two that need very little.

Every forecaster here has the same shape: it takes the history up to
and including the forecast origin (a panel, months down, series
across) and a horizon h, and returns one number per series for the
month h after the origin. The backtest calls them all the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive(history: pd.DataFrame, h: int) -> pd.Series:
    """Next month, and every month after, will be like the last one."""
    return history.iloc[-1]


def seasonal_naive(history: pd.DataFrame, h: int,
                   season: int = 12) -> pd.Series:
    """The same month last year. Missing if there was no last year."""
    target = history.index[-1] + h
    if target - season not in history.index:
        return pd.Series(np.nan, index=history.columns)
    return history.loc[target - season]


def moving_average(history: pd.DataFrame, h: int,
                   window: int = 3) -> pd.Series:
    """The average of the last `window` months."""
    return history.iloc[-window:].mean()


def simple_smoothing(history: pd.DataFrame, h: int,
                     alpha: float = 0.3) -> pd.Series:
    """Exponential smoothing: a level that moves alpha of the way
    towards each new month. The forecast is the last level."""
    level = history.iloc[0].to_numpy(float)
    for y in history.iloc[1:].to_numpy(float):
        level = np.where(np.isnan(level), y,
                         level + alpha * (y - level))
    return pd.Series(level, index=history.columns)


def seasonal_smoothing(history: pd.DataFrame, h: int,
                       alpha: float = 0.3, gamma: float = 0.2,
                       season: int = 12) -> pd.Series:
    """Holt-Winters without a trend: a smoothed level times a smoothed
    seasonal index for each calendar month. The first year sets the
    indices; each later month updates its own by gamma. With less
    than a year of history there is nothing to smooth, and it falls
    back to the same month last year."""
    Y = history.to_numpy(float)
    if len(Y) < season:
        return seasonal_naive(history, h, season)
    first = Y[:season]
    index = first / first.mean(axis=0)            # one per month
    level = first.mean(axis=0)
    for t in range(season, len(Y)):
        i = t % season
        new_level = level + alpha * (Y[t] / index[i] - level)
        index[i] = index[i] + gamma * (Y[t] / new_level - index[i])
        level = new_level
    target = len(Y) - 1 + h
    return pd.Series(level * index[target % season],
                     index=history.columns)


def smoothing_path(Y: np.ndarray, horizons, alpha: float = 0.3,
                   gamma: float = 0.2, season: int = 12) -> dict:
    """seasonal_smoothing's forecast from every origin at once, in one
    pass over the months: path[h][i] is the forecast made at month i
    for month i + h. The model in model.py uses it as a feature."""
    n = len(Y)
    path = {h: np.full(Y.shape, np.nan) for h in horizons}
    for i in range(min(season - 1, n)):      # under a year: last year
        for h in horizons:
            if 0 <= i + h - season:
                path[h][i] = Y[i + h - season]
    if n < season:
        return path
    index = Y[:season] / Y[:season].mean(axis=0)
    level = Y[:season].mean(axis=0)
    for i in range(season - 1, n):
        if i >= season:
            k = i % season
            new_level = level + alpha * (Y[i] / index[k] - level)
            index[k] = index[k] + gamma * (Y[i] / new_level - index[k])
            level = new_level
        for h in horizons:
            path[h][i] = level * index[(i + h) % season]
    return path
