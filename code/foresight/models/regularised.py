"""
Chapter 7's logistic regression with a penalty on the size of its
weights: ridge (L2) or lasso (L1). Chapter 9 writes it.

    python -m foresight.models.regularised     choose, then evaluate

The strength is chosen by a rolling backtest over training cohorts
whose outcomes were all on record by the first validation mark, so the
choice never sees a validation label and never uses the gap. The
validation cohorts are then read once, to confirm, with the rule and
v0.3 scored on the same rows.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foresight.config import ROOT, SEED
from foresight.data.build_table import TABLE
from foresight.evaluate import (SPLITS, auc, backtest, bootstrap,
                                difference, evaluate, hits_at_k,
                                interval, measure, report)
from foresight.models.linear import Standardiser
from foresight.models.logistic import (CALLS, RenewalRisk, features,
                                       log_loss)

# Cohorts ending July 2023 to April 2024: every one has at least four
# months of known outcomes before its mark, and every outcome among
# them was on record by 2 May 2024, the first validation cohort's mark.
TUNE = ("2023-07-01", "2024-04-30")
PENALTIES = ("l2", "l1")
STRENGTHS = (0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
             0.02)
REPORT = ROOT / "docs" / "foresight-evaluation-regularised.md"


class RegularisedRisk:
    """Standardise, then a logistic regression that minimises

        mean log loss + strength * sum of w**2      (penalty "l2")
        mean log loss + strength * sum of |w|       (penalty "l1")

    The strength is per contract, so the same number means the same
    leash whether the model learns from 3,000 rows or 6,000. Strength
    0 is v0.3's model. `drop` names column prefixes to leave out."""

    def __init__(self, penalty: str = "l2", strength: float = 0.001,
                 drop: tuple = ()):
        self.penalty, self.strength, self.drop = penalty, strength, drop

    def columns(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = features(rows)
        return X[[c for c in X.columns
                  if not c.startswith(tuple(self.drop))]]

    def fit(self, rows: pd.DataFrame):
        X = self.columns(rows)
        self.columns_ = list(X.columns)
        self.scaler_ = Standardiser().fit(X.to_numpy())
        n = len(rows)
        if self.strength == 0:
            m = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
        elif self.penalty == "l2":
            # scikit-learn minimises C * total loss + (1/2) sum w**2.
            m = LogisticRegression(C=1 / (2 * self.strength * n),
                                   tol=1e-10, max_iter=10_000)
        else:
            # C * total loss + sum |w|. liblinear also penalises the
            # intercept; scaling it by 100 makes that penalty tiny.
            m = LogisticRegression(l1_ratio=1, solver="liblinear",
                                   C=1 / (self.strength * n), tol=1e-8,
                                   intercept_scaling=100,
                                   max_iter=10_000, random_state=SEED)
        self.model_ = m.fit(self.scaler_.transform(X.to_numpy()),
                            rows.not_renewed.to_numpy())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Each contract's chance of not renewing."""
        Z = self.scaler_.transform(self.columns(rows).to_numpy())
        return self.model_.predict_proba(Z)[:, 1]

    def weights(self) -> pd.Series:
        """One weight per standardised column; lasso's zeros are 0.0."""
        return pd.Series(self.model_.coef_[0], index=self.columns_)


def maker(penalty: str, strength: float, drop: tuple = ()):
    """A make_model for backtest(): a new model with these settings."""
    return lambda: RegularisedRisk(penalty, strength, drop)


# ------------------------------------------------ choosing the strength
def tune(table: pd.DataFrame, penalties=PENALTIES,
         strengths=STRENGTHS, first: str = TUNE[0],
         last: str = TUNE[1], k: int = CALLS) -> pd.DataFrame:
    """Backtest every setting on the tuning cohorts: one row each."""
    rows = []
    for penalty in penalties:
        for s in strengths:
            scored = backtest(table, first, last, maker(penalty, s))
            y = scored.not_renewed.to_numpy()
            p = scored.model.to_numpy()
            hits = sum(hits_at_k(c.not_renewed, c.model,
                                 c.contract_id, k)
                       for _, c in scored.groupby("moment"))
            rows.append({"penalty": penalty, "strength": s,
                         "log loss": log_loss(y, p), "auc": auc(y, p),
                         "hits": hits})
    return pd.DataFrame(rows)


def choose(results: pd.DataFrame) -> dict:
    """The setting with the lowest log loss on the tuning cohorts."""
    best = results.loc[results["log loss"].idxmin()]
    return {"penalty": best.penalty, "strength": float(best.strength)}


# ------------------------------------------------ judging it
def head_to_head(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Two backtests of the same cohorts as one frame that bootstrap()
    reads: a's score as "model", b's as "rule". Same rows, same draws,
    so the difference is paired."""
    assert (a.contract_id.to_numpy() == b.contract_id.to_numpy()).all()
    return a.assign(rule=b.model.to_numpy())


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a minus b on precision at capacity, AUC and log loss, each with
    its paired 95% interval."""
    both = head_to_head(a, b)
    draws = bootstrap(both)
    point = measure(both)
    out = {}
    for what in ("precision", "auc"):
        d = point[what]["model"] - point[what]["rule"]
        out[what] = (d, *interval(difference(draws, what)))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    table = pd.read_parquet(TABLE)
    best = choose(tune(table))
    print(f"Chosen on {TUNE[0]} to {TUNE[1]}: {best['penalty']},"
          f" strength {best['strength']:g}\n")
    make = maker(best["penalty"], best["strength"])
    ev = evaluate("validation", table, make_model=make)
    print(report(ev, REPORT,
                 "Foresight evaluation: regularised renewal risk"))
    v03 = backtest(table, *SPLITS["validation"], RenewalRisk)
    c = compare(ev["scored"], v03)
    d, lo, hi = c["precision"]
    print(f"\nRegularised minus v0.3, paired\n  precision at {CALLS}"
          f"   {d * 100:+.1f} points ({lo * 100:+.1f} to"
          f" {hi * 100:+.1f})")
    d, lo, hi = c["auc"]
    print(f"  AUC               {d:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"\nWritten to {REPORT}")


if __name__ == "__main__":
    main()
