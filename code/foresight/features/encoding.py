"""
Target encoding: a category replaced by the share of its rows that
left. Chapter 12 writes it, twice, because the first way leaks.

    encode(cats, y, apply_to)      the rate in each category, learned
                                   from (cats, y), applied to apply_to
    out_of_fold(cats, y, groups)   each training row's rate learned
                                   from the other folds' rows only

Both shrink a small category's rate towards the overall rate: with
`weight` 20, a category of 20 rows is half its own rate and half the
table's. A category never seen gets the overall rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.config import SEED, rng

WEIGHT = 20
FOLDS = 5


def encode(cats: pd.Series, y: pd.Series, apply_to: pd.Series,
           weight: float = WEIGHT) -> np.ndarray:
    """Each value of `apply_to` as the smoothed leaving rate of its
    category in (cats, y)."""
    prior = float(np.mean(y))
    stats = pd.DataFrame({"c": np.asarray(cats), "y": np.asarray(y)})
    g = stats.groupby("c").y.agg(["sum", "size"])
    rate = (g["sum"] + weight * prior) / (g["size"] + weight)
    return (pd.Series(np.asarray(apply_to)).map(rate)
            .fillna(prior).to_numpy(dtype=float))


def out_of_fold(cats: pd.Series, y: pd.Series, groups: pd.Series,
                folds: int = FOLDS, weight: float = WEIGHT,
                seed: int = SEED) -> np.ndarray:
    """Training rows encoded without their own labels. Rows are dealt
    into folds by `groups` (the account), so no account's outcomes
    help encode its own rows; each fold is encoded from the others."""
    groups = np.asarray(groups)
    ids = np.unique(groups)
    fold_of = dict(zip(ids, rng(seed).integers(0, folds, len(ids))))
    fold = np.array([fold_of[a] for a in groups])
    cats, y = np.asarray(cats), np.asarray(y)
    out = np.empty(len(y))
    for f in range(folds):
        held = fold == f
        out[held] = encode(pd.Series(cats[~held]), pd.Series(y[~held]),
                           pd.Series(cats[held]), weight)
    return out
