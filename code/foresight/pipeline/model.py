"""
v0.6's renewal model as one scikit-learn object. Chapter 21 writes it.

Chapter 16's model is v0.5's lasso (models/featured.py) inside
Chapter 14's Calibrated (decide.py), and its preprocessing already
travels with it: features() and a Standardiser run inside its fit()
and predict_proba(). This module builds the same model from
scikit-learn's parts, so that every step has a name, can be looked
at, and is saved, loaded and checked as one object:

    prepare()   a ColumnTransformer: each table column to numbers
    Lasso       logistic regression, strength per contract (Ch 9)
    OutOfTime   Platt's map learned on the latest cohorts (Ch 14)

build() puts them together. Listing 21/02 checks that it scores the
validation cohorts as Chapter 16's model does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

from foresight.config import SEED
from foresight.decide import Platt

# The first category of each is the one the intercept carries, as in
# Chapter 7. A category not listed is an error, not a zero.
SEGMENTS = ["Small business", "Enterprise", "Mid-market",
            "Public sector"]
REGIONS = ["Southwest", "Midwest", "Northeast", "Southeast", "West"]
INPUTS = ["days_since_order", "orders_90d", "orders_prev_90d",
          "spend_365", "tickets_90d", "tenure_days", "segment",
          "region", "term_months", "legacy_terms", "discount_pct"]


def as_float(X: pd.DataFrame) -> pd.DataFrame:
    """Numbers as floats; a missing whole number becomes NaN."""
    return X.astype(float)


def numbers(fill=None):
    """A column of numbers, with its gaps filled by `fill`."""
    steps = [FunctionTransformer(as_float,
                                 feature_names_out="one-to-one")]
    if fill is not None:
        steps.append(SimpleImputer(strategy="constant",
                                   fill_value=fill))
    return make_pipeline(*steps)


def category(values: list[str]):
    """One column per value but the first. A gap counts as the first,
    as Chapter 7's features() counted it."""
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=values[0]),
        OneHotEncoder(categories=[values], drop="first",
                      sparse_output=False))


def prepare(no_order: float = 365) -> ColumnTransformer:
    """Chapter 7's sixteen numbers, one named step per column."""
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    return ColumnTransformer([
        ("gap", numbers(fill=no_order), ["days_since_order"]),
        ("orders", numbers(), ["orders_90d", "orders_prev_90d"]),
        ("spend", make_pipeline(numbers(), log), ["spend_365"]),
        ("as_is", numbers(), ["tickets_90d", "tenure_days",
                              "term_months", "legacy_terms"]),
        ("discount", numbers(fill=0), ["discount_pct"]),
        ("segment", category(SEGMENTS), ["segment"]),
        ("region", category(REGIONS), ["region"]),
    ], verbose_feature_names_out=False)


class Lasso(ClassifierMixin, BaseEstimator):
    """Chapter 9's lasso. Its strength is per contract, so the penalty
    scikit-learn is given, C, depends on how many rows it fits."""

    def __init__(self, strength: float = 0.002):
        self.strength = strength

    def fit(self, X, y):
        C = 1 / (self.strength * len(y))
        self.model_ = LogisticRegression(
            l1_ratio=1, solver="liblinear", C=C, tol=1e-8,
            intercept_scaling=100, max_iter=10_000,
            random_state=SEED).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class OutOfTime(ClassifierMixin, BaseEstimator):
    """Chapter 14's Calibrated for any scikit-learn classifier: learn
    Platt's map from a copy fitted before the latest `months` cohorts
    and scored on them, then refit on every row."""

    def __init__(self, estimator, months: int = 3):
        self.estimator, self.months = estimator, months

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        first = np.sort(X.moment.unique())[-self.months]
        held = (X.moment >= first).to_numpy()
        early = (X.end_date < first).to_numpy()
        if len(np.unique(y[early])) < 2:
            raise ValueError("too little history before the held"
                             " months")
        scores = (clone(self.estimator).fit(X[early], y[early])
                  .predict_proba(X[held])[:, 1])
        self.map_ = Platt().fit(scores, y[held])
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        p = self.map_(self.estimator_.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def build(strength: float = 0.002, months: int = 3,
          no_order: float = 365) -> OutOfTime:
    """v0.6: prepare, standardise, lasso; calibrated out of time."""
    lasso = Pipeline([("prepare", prepare(no_order)),
                      ("scale", StandardScaler()),
                      ("lasso", Lasso(strength))])
    return OutOfTime(lasso, months)


def weights(fitted: OutOfTime) -> pd.Series:
    """The lasso's weight on each standardised column, by name."""
    inner = fitted.estimator_
    names = inner.named_steps["prepare"].get_feature_names_out()
    coef = inner.named_steps["lasso"].model_.coef_[0]
    return pd.Series(coef, index=names)


class ForBacktest:
    """A built model in the shape evaluate.backtest() expects:
    fit(rows) and predict_proba(rows) giving one chance a row."""

    def __init__(self, model):
        self.model = model

    def fit(self, rows: pd.DataFrame):
        self.model = clone(self.model).fit(rows, rows.not_renewed)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(rows)[:, 1]


def maker(model):
    """A make_model for backtest(): a fresh copy of `model`."""
    return lambda: ForBacktest(model)
