"""
Forecasting as regression: one LightGBM model for every series.

Each training row is a (series, origin, horizon) triple. Its features
are what was known at the origin, and its target is the month h after
it. Nothing is measured after the origin, whatever the horizon: that
is the rule that keeps the backtest honest.

Every row is put on its own series' scale, the average of the three
months up to the origin, so that a series selling 300 units a month
and one selling 7,000 teach the model the same shapes. The model
predicts the ratio of the target to that scale.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.forecast.backtest import HORIZONS, TARGETS
from foresight.forecast.baselines import smoothing_path

FEATURES = ["h", "month", "lag0", "lag1", "lag2", "mean6", "mean12",
            "last_year", "last_year_step", "smoothed", "category",
            "region"]
PARAMS = {"n_estimators": 150, "learning_rate": 0.1,
          "num_leaves": 15, "min_child_samples": 20}


def lag_rows(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Features known at each origin, and the target h months on."""
    Y = panel.to_numpy(float)
    months = panel.index
    ids = list(panel.columns)
    blank = np.full(Y.shape[1], np.nan)
    smooth = smoothing_path(Y, horizons)       # one pass, every origin

    def mean(a, b):                             # months a..b inclusive
        return Y[a:b + 1].mean(axis=0) if a >= 0 else blank

    out = []
    for i in range(2, len(months)):
        scale = mean(i - 2, i)                  # NaN if any is NaN
        for h in horizons:
            j = i + h - 12                      # the target, a year ago
            ago = Y[j] if j >= 0 else blank
            out.append(pd.DataFrame({
                "series": ids, "origin": months[i],
                "target": months[i] + h, "h": h,
                "month": (months[i] + h).month, "scale": scale,
                "lag0": Y[i] / scale, "lag1": Y[i - 1] / scale,
                "lag2": Y[i - 2] / scale,
                "mean6": mean(i - 5, i) / scale,
                "mean12": mean(i - 11, i) / scale,
                "last_year": ago / scale,
                "last_year_step": ago / mean(i - 14, i - 12),
                "smoothed": smooth[h][i] / scale,
                "actual": Y[i + h] if i + h < len(months) else blank}))
    rows = pd.concat(out, ignore_index=True)
    for k, name in enumerate(panel.columns.names):
        rows[name] = [s[k] for s in rows.series]
    rows = rows[np.isfinite(rows.scale) & (rows.scale > 0)]
    for name in ("category", "region"):
        rows[name] = rows[name].astype("category")
    return rows.reset_index(drop=True)


class GlobalModel:
    """LightGBM on scaled rows. alpha=None fits the median (absolute
    error); alpha=0.1 or 0.9 fits that quantile instead."""

    def __init__(self, alpha: float | None = None, **params):
        objective = ({"objective": "l1"} if alpha is None else
                     {"objective": "quantile", "alpha": alpha})
        self.model = lgb.LGBMRegressor(
            **objective, **{**PARAMS, **params},
            **LIGHTGBM_DETERMINISTIC)

    def fit(self, rows: pd.DataFrame) -> "GlobalModel":
        rows = rows[rows.actual.notna()]
        self.model.fit(rows[FEATURES], rows.actual / rows.scale)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        ratio = self.model.predict(rows[FEATURES])
        return ratio * rows.scale.to_numpy()


def training_rows(panel: pd.DataFrame, extra: tuple = (),
                  horizons=HORIZONS) -> tuple:
    """The rows to forecast (panel's) and the rows to learn from
    (panel's, plus any `extra` panels: more series of the same kind)."""
    rows = lag_rows(panel, horizons)
    train = pd.concat([rows] + [lag_rows(p, horizons) for p in extra],
                      ignore_index=True)
    for name in ("category", "region"):
        train[name] = train[name].astype(rows[name].dtype)
    return rows, train


def forecast_from(origin, rows: pd.DataFrame, train: pd.DataFrame,
                  alphas=(None,), **params) -> pd.DataFrame:
    """Stand at `origin`: fit on rows whose target is on record by
    then, and forecast every row made at that origin."""
    seen = train[train.target <= origin]
    test = rows[rows.origin == origin].copy()
    for alpha in alphas:
        model = GlobalModel(alpha, **params).fit(seen)
        test[_column(alpha)] = model.predict(test)
    return test


def backtest_model(panel: pd.DataFrame, extra: tuple = (),
                   alphas=(None,), horizons=HORIZONS,
                   targets=TARGETS, **params) -> pd.DataFrame:
    """The expanding-window backtest for the global model: refit at
    every origin, forecast the targets. One column per alpha."""
    rows, train = training_rows(panel, extra, horizons)
    rows = rows[rows.target.isin(targets)]
    out = [forecast_from(origin, rows, train, alphas, **params)
           for origin in sorted(rows.origin.unique())]
    keep = ["series", "h", "origin", "target", "actual",
            *[_column(a) for a in alphas]]
    return pd.concat(out, ignore_index=True)[keep]


def _column(alpha) -> str:
    return "forecast" if alpha is None else f"q{round(alpha * 100):02d}"
