"""
Isolation forests over ticket counts, and the path lengths that make
them work.

A forest isolates a row by splitting at random: pick a column, pick a
value between that column's smallest and largest, repeat until the row
is alone. A row that is unusual is cut off in few splits. The score is
the path length, averaged over many random trees and compared with the
path length expected for an ordinary row.

What a row is decides what the forest can see. Foresight's rows are
series-days: one series (a supplier or a category) on one day, with
two columns, how surprising today's count is and how surprising the
last seven days are, each measured against that series' own trailing
baseline. A whole day with one column per series is the obvious
alternative, and Chapter 18 shows why it is the weaker one.

The forest is refitted at the start of every month on the year before
it, so each day is scored by a model that never saw it. A score is
turned into a rank: the share of the training year's rows that scored
at least as high. An alert is a rank at or below `q`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from foresight.config import SEED

EULER = 0.5772156649


def c(n):
    """Average path length of an unsuccessful search in a binary tree
    of n points: what a leaf still holding n points adds to a path."""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = (2 * (np.log(n[big] - 1) + EULER)
                - 2 * (n[big] - 1) / n[big])
    return out


def path_lengths(model: IsolationForest, X) -> np.ndarray:
    """Each row's path length averaged over the forest's trees: how many
    random splits it took to isolate the row."""
    X = np.asarray(X, dtype=float)
    total = np.zeros(len(X))
    for tree, cols in zip(model.estimators_,
                          model.estimators_features_):
        t = tree.tree_
        depth = np.zeros(t.node_count)
        for node in range(t.node_count):     # parents come first
            for child in (t.children_left[node],
                          t.children_right[node]):
                if child != -1:
                    depth[child] = depth[node] + 1
        leaf = tree.apply(X[:, cols].astype(np.float32))
        total += depth[leaf] + c(t.n_node_samples[leaf])
    return total / len(model.estimators_)


def surprise(counts: pd.DataFrame, window: int = 56):
    """Today's count and the last seven days' total, each as standard
    deviations above what the `window` days before today lead one to
    expect. Counts are treated as Poisson, so the spread is the square
    root of the expected count."""
    mean = counts.shift(1).rolling(window, min_periods=window).mean()
    lam = np.maximum(mean, 1.0 / window)
    today = (counts - mean) / np.sqrt(lam)
    week = counts.rolling(7).sum()
    past = (counts.shift(7).rolling(window, min_periods=window).mean())
    lam7 = 7 * np.maximum(past, 1.0 / window)
    last7 = (week - 7 * past) / np.sqrt(lam7)
    return today, last7


def series_days(counts: pd.DataFrame, window: int = 56) -> pd.DataFrame:
    """One row per series per day, with its two surprises."""
    today, last7 = surprise(counts, window)
    rows = pd.DataFrame({"z_today": today.stack(),
                         "z_week": last7.stack()}).dropna()
    rows.index.names = ["day", "series"]
    return rows


def fit_forest(X, seed: int = SEED,
               n_estimators: int = 200) -> IsolationForest:
    return IsolationForest(n_estimators=n_estimators,
                           random_state=seed).fit(np.asarray(X))


def rolling_ranks(rows: pd.DataFrame, first: str,
                  train_days: int = 365,
                  seed: int = SEED) -> pd.DataFrame:
    """Score every row from `first` on with a forest fitted on the rows
    of the `train_days` before its month. Rows are indexed by day, or by
    (day, series). Returns the score (higher is stranger) and the rank:
    the share of training rows scoring at least as high."""
    days = rows.index.get_level_values(0)
    out = []
    for m in pd.date_range(first, days.max(), freq="MS"):
        train = rows[(days < m)
                     & (days >= m - pd.Timedelta(days=train_days))]
        month = rows[(days >= m) & (days < m + pd.offsets.MonthBegin())]
        model = fit_forest(train.to_numpy(), seed)
        base = np.sort(-model.score_samples(train.to_numpy()))
        score = -model.score_samples(month.to_numpy())
        above = len(base) - np.searchsorted(base, score, side="left")
        out.append(pd.DataFrame({"score": score,
                                 "rank": above / len(base)},
                                index=month.index))
    return pd.concat(out)


def forest_alerts(ranks: pd.DataFrame, q: float) -> pd.Series:
    """True where a row is stranger than all but a share q of its
    training year."""
    return ranks["rank"] <= q
