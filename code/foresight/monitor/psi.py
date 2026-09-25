"""
The population stability index: how far a column's distribution this
month has moved from the one the model learned from. Chapter 24.

Cut the reference rows into bins, count the share of reference rows and
of new rows in each bin, and add up, bin by bin,

    (new share - reference share) * ln(new share / reference share)

Both halves of each term have the same sign, so every bin adds
something, and a bin whose share doubled adds about as much as one
whose share halved. The bins are fixed on the reference rows, the rows
the model learned from, and never recomputed from the new rows: a bin
edge that moves with the data measures the edge, not the data.

A category is its own bin. A missing value is its own bin too, because
"no order on record" is a value the model reads. A bin that is empty on
one side would make the logarithm infinite, so shares are floored at
FLOOR before the sum.

The two thresholds are the credit-scoring industry's rule of thumb, not
a result of this book: below 0.10 nothing has moved that matters, above
0.25 the population is not the one the model was fitted on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WATCH, ACT = 0.10, 0.25
FLOOR = 1e-4            # a share of 0.01%: an empty bin, nearly
BINS = 10
CATEGORICAL = ("segment", "region", "term_months", "legacy_terms",
               "discount_pct")


def edges(reference, bins: int = BINS) -> np.ndarray:
    """Interior cut points at the reference's deciles (or other
    quantiles), missing values ignored, repeated cuts dropped."""
    x = pd.Series(reference, dtype="Float64").dropna().to_numpy(float)
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(q)


def shares(values, cuts=None) -> pd.Series:
    """The share of `values` in each bin: numbered bins for numbers
    (bin i holds values up to cuts[i]), the value itself for a
    category, and "missing" for a gap."""
    s = pd.Series(values)
    gap = s.isna().to_numpy()
    if cuts is None:
        labels = s.astype(object).where(~gap, "missing").astype(str)
    else:
        x = s.astype("Float64").fillna(0).to_numpy(float)
        labels = pd.Series(np.searchsorted(cuts, x, side="left"),
                           dtype=object).where(~gap, "missing")
        labels = labels.astype(str)
    return labels.value_counts(normalize=True).sort_index()


def psi(reference: pd.Series, new: pd.Series,
        floor: float = FLOOR) -> float:
    """PSI between two sets of bin shares (from shares())."""
    both = reference.index.union(new.index)
    a = np.maximum(reference.reindex(both).fillna(0).to_numpy(), floor)
    b = np.maximum(new.reindex(both).fillna(0).to_numpy(), floor)
    return float(((b - a) * np.log(b / a)).sum())


class Baseline:
    """Bins and shares fixed on the reference rows, once: every later
    month is compared with the same bins."""

    def __init__(self, reference: pd.DataFrame, columns,
                 bins: int = BINS):
        self.columns = list(columns)
        self.cuts = {c: None if c in CATEGORICAL
                     else edges(reference[c], bins)
                     for c in self.columns}
        self.shares = {c: shares(reference[c], self.cuts[c])
                       for c in self.columns}

    def column(self, rows: pd.DataFrame, c: str) -> float:
        return psi(self.shares[c], shares(rows[c], self.cuts[c]))

    def table(self, rows: pd.DataFrame, by: str = "moment"
              ) -> pd.DataFrame:
        """PSI of every column for every group of rows (every cohort,
        by default): one row per group, one column per column."""
        out = {k: {c: self.column(g, c) for c in self.columns}
               for k, g in rows.groupby(by)}
        return pd.DataFrame(out).T[self.columns]


def level(value: float) -> str:
    """The rule of thumb in words."""
    if value >= ACT:
        return "moved"
    return "watch" if value >= WATCH else "stable"
