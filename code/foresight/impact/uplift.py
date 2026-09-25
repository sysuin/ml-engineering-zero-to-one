"""
A two-model uplift sketch. Chapter 25 draws it; Foresight does not
ship it.

An uplift model ranks accounts by how much a call changes their chance
of leaving, not by the chance itself. The simplest kind fits two risk
models on a randomised test: one on the held-out accounts (what
happens without a call) and one on the called accounts (what happens
with one). An account's uplift is the first model's chance minus the
second's. It needs calls that were assigned at random, across the
range of accounts it will choose from; without them the two models
learn who the team chose to call, not what calling did.

Here each model is a logistic regression on the risk score's log-odds
and its square, so the estimated uplift can rise and then fall with
risk, as the persuadables argument says it might.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def features(chance) -> np.ndarray:
    """The risk score's log-odds, and its square."""
    p = np.clip(np.asarray(chance, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    return np.column_stack([x, x ** 2])


class TwoModel:
    """Uplift as the difference between two risk models."""

    def fit(self, X, left, called) -> "TwoModel":
        called = np.asarray(called, dtype=bool)
        left = np.asarray(left)
        self.without_ = LogisticRegression().fit(X[~called],
                                                 left[~called])
        self.with_ = LogisticRegression().fit(X[called], left[called])
        return self

    def predict(self, X) -> np.ndarray:
        """Estimated fall in the chance of leaving if called."""
        return (self.without_.predict_proba(X)[:, 1]
                - self.with_.predict_proba(X)[:, 1])
