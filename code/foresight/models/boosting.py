"""
Gradient boosting for renewal risk. Chapter 11 writes it, and judges it
beside the rule, Chapter 7's logistic model, Chapter 9's lasso and
Chapter 10's forest on v0.4's leaderboard.

    python -m foresight.models.boosting      the leaderboard, validation

On that leaderboard it was level with the lasso and the logistic model,
and v0.4 ships the lasso, the simplest model nothing beat (Chapter 11
argues it). This model is the challenger, judged again when the table
gains new columns.

RenewalBooster has the same fit(rows) and predict_proba(rows) as every
Foresight model, so Chapter 8's backtest runs it unchanged. It reads the
table's columns as they are: gaps stay gaps, for LightGBM to route, and
segment and region stay categories.

The number of rounds is chosen inside fit(), by early stopping on the
last three months of outcomes in the rows it is given, never on the
cohorts it will be judged on. The rows it fits while watching are the
ones whose outcomes were on record at the watched cohorts' marks, the
same rule the backtest applies to it. Then it refits on every row with
the rounds it found.

Every other setting was fixed on the training period before any
validation cohort was scored by it (Chapter 11): the learning rate by
the rounds each rate needed, the tree size and leaf size by the watched
log loss, and one monotonic constraint, on days since the last order,
because a longer gap must never lower an account's risk.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC

RATE = 0.03             # each tree's step, shrunk to 3%
LEAVES = 2              # leaves per tree: one question each, a stump
MIN_LEAF = 50           # contracts in the smallest leaf
MOST = 3000             # the most rounds early stopping may reach
PATIENCE = 200          # rounds without improvement before it stops
WATCH_MONTHS = 3        # the months of outcomes it watches
RISING = ("days_since_order",)      # risk may not fall as these rise

NUMERIC = ["days_since_order", "orders_90d", "orders_prev_90d",
           "spend_365", "tickets_90d", "tenure_days", "term_months",
           "legacy_terms", "discount_pct"]
SEGMENTS = ["Enterprise", "Mid-market", "Public sector",
            "Small business"]
REGIONS = ["Midwest", "Northeast", "Southeast", "Southwest", "West"]


def columns(rows: pd.DataFrame) -> pd.DataFrame:
    """The table's columns as LightGBM reads them. No fills: a contract
    with no order before its mark, or on legacy terms with no discount,
    keeps its gap. The categories are fixed, so every fit sees the same
    codes whichever categories its rows happen to contain."""
    X = rows[NUMERIC].astype(float)
    X["segment"] = pd.Categorical(rows.segment, categories=SEGMENTS)
    X["region"] = pd.Categorical(rows.region, categories=REGIONS)
    return X


def watch_split(rows: pd.DataFrame, months: int = WATCH_MONTHS):
    """The last `months` monthly cohorts to watch, and the rows whose
    outcomes were on record at the first watched mark to fit on."""
    marks = np.sort(rows.moment.unique())
    watch = rows[rows.moment >= marks[-months]]
    fit = rows[rows.end_date < marks[-months]]
    if fit.empty or fit.not_renewed.nunique() < 2:
        raise ValueError("too little history before the watched months")
    return fit, watch


class RenewalBooster:
    """LightGBM, stopped early on the latest outcomes, then refitted."""

    def __init__(self, rate: float = RATE, leaves: int = LEAVES,
                 min_leaf: int = MIN_LEAF, rising=RISING,
                 most: int = MOST, patience: int = PATIENCE,
                 months: int = WATCH_MONTHS):
        self.rate, self.leaves, self.min_leaf = rate, leaves, min_leaf
        self.rising, self.most, self.patience = rising, most, patience
        self.months = months

    def params(self, rounds: int) -> dict:
        """Every setting LightGBM is given."""
        names = NUMERIC + ["segment", "region"]
        return {**LIGHTGBM_DETERMINISTIC, "n_estimators": rounds,
                "learning_rate": self.rate, "num_leaves": self.leaves,
                "min_child_samples": self.min_leaf,
                "monotone_constraints": [int(c in self.rising)
                                         for c in names]}

    def fit(self, rows: pd.DataFrame):
        fit, watch = watch_split(rows, self.months)
        stop = lgb.early_stopping(self.patience, verbose=False)
        stopper = LGBMClassifier(**self.params(self.most)).fit(
            columns(fit), fit.not_renewed, eval_X=columns(watch),
            eval_y=watch.not_renewed, callbacks=[stop])
        self.rounds_ = int(stopper.best_iteration_)
        self.watched_ = stopper.evals_result_["valid_0"][
            "binary_logloss"]           # the watched loss, every round
        self.model_ = LGBMClassifier(**self.params(self.rounds_)).fit(
            columns(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Each contract's chance of not renewing."""
        return self.model_.predict_proba(columns(rows))[:, 1]


def contenders() -> dict:
    """v0.4's leaderboard: four models, in order of complexity. The
    lasso's strength is the one Chapter 9 chose on training cohorts."""
    from foresight.models.forest import RenewalForest
    from foresight.models.logistic import RenewalRisk
    from foresight.models.regularised import maker
    return {"logistic": RenewalRisk, "lasso": maker("l1", 0.002),
            "forest": RenewalForest, "boosting": RenewalBooster}


def main() -> None:
    from foresight.leaderboard import report, run
    print(report(run(contenders())))


if __name__ == "__main__":
    main()
