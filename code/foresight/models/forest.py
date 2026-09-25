"""
A random forest for renewal risk. Chapter 10 writes it, to be judged
beside Chapter 7's logistic model and the rule by Chapter 8's backtest.

    python -m foresight.models.forest       the leaderboard, validation

RenewalForest has the same fit(rows) and predict_proba(rows) as
RenewalRisk, so evaluate(make_model=RenewalForest) runs it unchanged.
It sees Chapter 7's sixteen columns as they are: a tree asks only
whether a value is above or below a cut, so scaling changes nothing.

Its settings were fixed before any validation cohort was scored by it:
the smallest leaf by out-of-bag AUC on the training split (Chapter 10),
the columns per split at the library's default for a classifier, and
enough trees that more no longer move the out-of-bag AUC's third
decimal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED
from foresight.models.logistic import features

TREES = 300             # trees in the forest
MIN_LEAF = 50           # contracts in the smallest leaf, chosen by OOB
COLUMNS = "sqrt"        # columns offered at each split: 4 of 16


class RenewalForest:
    """Many deep trees on bootstrap samples, a random few columns at
    each split, their leaf rates averaged."""

    def __init__(self, trees: int = TREES, min_leaf: int = MIN_LEAF,
                 columns=COLUMNS, seed: int = SEED):
        self.trees, self.min_leaf = trees, min_leaf
        self.columns, self.seed = columns, seed

    def fit(self, rows: pd.DataFrame):
        X = features(rows)
        self.columns_ = list(X.columns)
        self.model_ = RandomForestClassifier(
            n_estimators=self.trees, min_samples_leaf=self.min_leaf,
            max_features=self.columns, random_state=self.seed,
            n_jobs=1)               # one thread: the same trees always
        self.model_.fit(X.to_numpy(), rows.not_renewed.to_numpy())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Each contract's chance of not renewing: the average, over
        the trees, of the leaver rate in the leaf it lands in."""
        X = features(rows).to_numpy()
        return self.model_.predict_proba(X)[:, 1]

    def importances(self) -> pd.Series:
        """Impurity importance: each column's share of the Gini the
        trees removed, largest first. Measured on the training rows,
        so it rewards a column that fitted noise; Chapter 10 shows
        how badly, and Chapter 16 measures importance on held-out
        rows instead."""
        return pd.Series(self.model_.feature_importances_,
                         index=self.columns_).sort_values(
                             ascending=False)


def main() -> None:
    from foresight.leaderboard import report, run
    print(report(run()))


if __name__ == "__main__":
    main()
