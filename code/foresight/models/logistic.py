"""
Foresight's first renewal-risk model: logistic regression on Chapter 4's
table, scored beside the days-since-last-order rule every time it is
scored. Chapter 7 writes it.

    python -m foresight.models.logistic        train, then compare

The model is fitted on the training split and judged on validation, by
precision among the top 40 of each monthly cohort. The test year is not
read. Chapter 8 replaces the fixed splits with a backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser

TRAIN = ("2023-01-01", "2024-06-30")
VALIDATION = ("2024-07-01", "2024-12-31")
CALLS = 40                      # retention calls per monthly cohort

# One column per category but the largest, which the intercept carries:
# Small business and the Southwest. The 12 training rows with no segment
# on record count with Small business.
SEGMENTS = ["Enterprise", "Mid-market", "Public sector"]
REGIONS = ["Midwest", "Northeast", "Southeast", "West"]


def load(first: str, last: str) -> pd.DataFrame:
    """Chapter 4's rows for contracts ending in [first, last]."""
    rows = pd.read_parquet(TABLE)
    rows = rows[rows.end_date.between(first, last)]
    return rows.reset_index(drop=True)


def features(rows: pd.DataFrame) -> pd.DataFrame:
    """Every column as a number, one row per contract."""
    X = pd.DataFrame(index=rows.index)
    # No order before the mark: a year, as in Chapter 6.
    gap = rows.days_since_order.astype(float)
    X["days_since_order"] = gap.fillna(365)
    X["orders_90d"] = rows.orders_90d.astype(float)
    X["orders_prev_90d"] = rows.orders_prev_90d.astype(float)
    X["log_spend_365"] = np.log1p(rows.spend_365)  # Chapter 5's scale
    X["tickets_90d"] = rows.tickets_90d.astype(float)
    X["tenure_days"] = rows.tenure_days.astype(float)
    X["term_months"] = rows.term_months.astype(float)
    # Legacy terms have no discount field: legacy_terms marks the gap,
    # and the fill is a placeholder its weight absorbs.
    X["legacy_terms"] = rows.legacy_terms.astype(float)
    X["discount_pct"] = rows.discount_pct.astype(float).fillna(0)
    for s in SEGMENTS:
        X[f"segment={s}"] = (rows.segment == s).astype(float)
    for r in REGIONS:
        X[f"region={r}"] = (rows.region == r).astype(float)
    return X


# ------------------------------------------------ the arithmetic
def sigmoid(z):
    """Squeeze any number into (0, 1)."""
    return 1 / (1 + np.exp(-z))


def log_loss(y, p, eps: float = 1e-15) -> float:
    """The mean of -log(p) for leavers and -log(1 - p) for the rest."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_gradient(X, y, w, b):
    """How the log loss changes with each weight and b."""
    error = sigmoid(X @ w + b) - y
    return X.T @ error / len(y), error.mean()


class RenewalRisk:
    """Standardise, then an unpenalised logistic regression."""

    def fit(self, rows: pd.DataFrame):
        X = features(rows)
        self.columns_ = list(X.columns)
        self.scaler_ = Standardiser().fit(X.to_numpy())
        self.model_ = LogisticRegression(C=np.inf, tol=1e-10,
                                         max_iter=10_000)
        self.model_.fit(self.scaler_.transform(X.to_numpy()),
                        rows.not_renewed.to_numpy())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Each contract's chance of not renewing."""
        Z = self.scaler_.transform(features(rows).to_numpy())
        return self.model_.predict_proba(Z)[:, 1]


# ------------------------------------------------ scoring a list
def days_since_rule(rows: pd.DataFrame) -> np.ndarray:
    """The baseline's score: the longest gap first, no order last."""
    return rows.days_since_order.astype(float).fillna(-1).to_numpy()


def top_of_each_cohort(rows: pd.DataFrame, score,
                       k: int = CALLS) -> pd.DataFrame:
    """The k highest-scoring contracts of every monthly cohort."""
    ranked = rows.assign(score=score).sort_values(
        ["moment", "score", "contract_id"],
        ascending=[True, False, True])
    return ranked.groupby("moment").head(k)


def at_capacity(rows: pd.DataFrame, score, k: int = CALLS) -> dict:
    """Leavers reached, precision and recall, pooled over cohorts."""
    top = top_of_each_cohort(rows, score, k)
    hits = int(top.not_renewed.sum())
    return {"calls": len(top), "leavers": hits,
            "precision": hits / len(top),
            "recall": hits / int(rows.not_renewed.sum())}


def compare(train: pd.DataFrame | None = None,
            rows: pd.DataFrame | None = None,
            k: int = CALLS) -> dict:
    """Fit on train, score the model and the rule on rows, and add a
    random and a perfect list for scale. Never one without the other."""
    train = load(*TRAIN) if train is None else train
    rows = load(*VALIDATION) if rows is None else rows
    model = RenewalRisk().fit(train)
    cohorts = rows.groupby("moment").not_renewed
    calls = cohorts.size().clip(upper=k)
    lucky = float((cohorts.mean() * calls).sum())  # random, expected
    best = int(cohorts.sum().clip(upper=k).sum())  # leavers first
    leavers = int(rows.not_renewed.sum())
    return {
        "model": at_capacity(rows, model.predict_proba(rows), k),
        "rule": at_capacity(rows, days_since_rule(rows), k),
        **{name: {"calls": int(calls.sum()), "leavers": n,
                  "precision": n / calls.sum(),
                  "recall": n / leavers}
           for name, n in (("random", lucky), ("perfect", best))},
    }


def main() -> None:
    result = compare()
    print(f"Validation, top {CALLS} of each monthly cohort")
    print(f"  {'':10}{'leavers':>9}{'precision':>11}{'recall':>9}")
    for name, s in result.items():
        print(f"  {name:10}{s['leavers']:>9.1f}"
              f"{s['precision']:>11.1%}{s['recall']:>9.1%}")


if __name__ == "__main__":
    main()
