"""
Foresight v0.3's renewal-risk model from Chapter 7: the arithmetic
checked against answers worked out another way, the features against
the rules Chapters 5 and 6 set, and the comparison against lists whose
scores are known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from foresight.config import rng
from foresight.models.linear import descend
from foresight.models.logistic import (RenewalRisk, at_capacity, compare,
                                       days_since_rule, features,
                                       log_loss, log_loss_gradient,
                                       sigmoid, top_of_each_cohort)


def table(n_per_cohort=60, cohorts=3, seed=0) -> pd.DataFrame:
    """A made-up renewals table with the Chapter 4 columns, in which a
    long gap since the last order really does raise the chance of
    leaving."""
    g = rng(seed)
    n = n_per_cohort * cohorts
    gap = g.integers(1, 200, n).astype(float)
    gap[:3] = np.nan                                  # never ordered
    legacy = (g.random(n) < 0.25).astype(int)
    z = -3 + 0.02 * np.nan_to_num(gap, nan=365) + g.normal(0, 0.5, n)
    moments = pd.date_range("2023-04-02", periods=cohorts, freq="MS")
    return pd.DataFrame({
        "contract_id": np.arange(n) + 1,
        "moment": np.repeat(moments, n_per_cohort),
        "days_since_order": pd.array(gap, dtype="Float64").astype("Int64"),
        "orders_90d": g.integers(0, 20, n),
        "orders_prev_90d": g.integers(0, 20, n),
        "spend_365": g.lognormal(9, 1, n),
        "tickets_90d": g.integers(0, 3, n),
        "tenure_days": g.integers(100, 4000, n),
        "segment": g.choice(["Small business", "Mid-market", "Enterprise",
                             "Public sector"], n),
        "region": g.choice(["Midwest", "Northeast", "Southeast",
                            "Southwest", "West"], n),
        "term_months": g.choice([12, 24], n),
        "legacy_terms": legacy,
        "discount_pct": pd.array(np.where(legacy == 1, np.nan,
                                          g.choice([0, 5, 10], n)),
                                 dtype="Float64").astype("Int64"),
        "not_renewed": (g.random(n) < sigmoid(z)).astype(int),
    })


def test_the_sigmoid_is_a_half_at_zero_and_symmetric():
    z = np.linspace(-8, 8, 33)
    assert sigmoid(0.0) == 0.5
    np.testing.assert_allclose(sigmoid(-z), 1 - sigmoid(z))
    assert ((sigmoid(z) > 0) & (sigmoid(z) < 1)).all()


def test_log_loss_punishes_a_confident_wrong_answer():
    assert log_loss(np.array([1]), np.array([0.5])) == pytest.approx(
        np.log(2))
    assert log_loss(np.array([1]), np.array([0.01])) > 4.6
    assert np.isfinite(log_loss(np.array([1]), np.array([0.0])))


def test_the_gradient_matches_a_numerical_one():
    g = rng()
    X, y = g.normal(size=(50, 3)), (g.random(50) < 0.3).astype(float)
    w, b, h = np.array([0.3, -0.2, 0.1]), -0.5, 1e-6

    def loss(w, b):
        return log_loss(y, sigmoid(X @ w + b))

    dw, db = log_loss_gradient(X, y, w, b)
    for i in range(3):
        e = np.eye(3)[i] * h
        assert dw[i] == pytest.approx(
            (loss(w + e, b) - loss(w - e, b)) / (2 * h), rel=1e-5)
    assert db == pytest.approx(
        (loss(w, b + h) - loss(w, b - h)) / (2 * h), rel=1e-5)


def test_descent_with_the_log_loss_gradient_matches_scikit_learn():
    g = rng()
    X = g.normal(size=(400, 3))
    y = (g.random(400) < sigmoid(X @ [1.0, -0.5, 0.0] - 1)).astype(float)
    w, b = descend(X, y, lr=1.0, steps=5000, gradient=log_loss_gradient)
    lib = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
    lib.fit(X, y)
    np.testing.assert_allclose(w, lib.coef_[0], atol=1e-5)
    assert b == pytest.approx(lib.intercept_[0], abs=1e-5)


def test_features_fill_the_gaps_and_keep_the_indicator():
    rows = table()
    X = features(rows)
    assert not X.isna().any().any()
    legacy = rows.legacy_terms == 1
    assert (X.discount_pct[legacy] == 0).all()
    assert (X.legacy_terms[legacy] == 1).all()
    assert (X.days_since_order[rows.days_since_order.isna()] == 365).all()
    assert "segment=Small business" not in X.columns   # the reference
    assert "region=Southwest" not in X.columns
    one_hot = X.filter(like="segment=").sum(axis=1)
    assert one_hot.max() == 1


def test_the_rule_ranks_the_longest_gap_first_and_no_order_last():
    rows = table()
    score = days_since_rule(rows)
    top = top_of_each_cohort(rows, score, k=5)
    for _, cohort in rows.assign(s=score).groupby("moment"):
        chosen = top[top.moment == cohort.moment.iloc[0]]
        assert chosen.days_since_order.min() >= cohort.s.nlargest(5).min()
    assert (score[rows.days_since_order.isna()] == -1).all()


def test_capacity_scores_pool_the_cohorts():
    rows = table()
    perfect = at_capacity(rows, rows.not_renewed.to_numpy(), k=10)
    per = rows.groupby("moment").not_renewed.sum().clip(upper=10)
    assert perfect["calls"] == 30
    assert perfect["leavers"] == per.sum()
    assert perfect["recall"] == pytest.approx(
        per.sum() / rows.not_renewed.sum())


def test_compare_reports_the_rule_beside_the_model_every_time():
    train, rows = table(seed=1), table(seed=2)
    result = compare(train, rows, k=10)
    assert set(result) == {"model", "rule", "random", "perfect"}
    for s in result.values():
        assert s["calls"] == 30
        assert 0 <= s["precision"] <= 1
    assert result["perfect"]["precision"] >= result["model"]["precision"]
    assert result["random"]["leavers"] == pytest.approx(
        rows.not_renewed.mean() * 30, rel=0.2)


def test_the_model_learns_the_planted_pattern():
    model = RenewalRisk().fit(table(seed=3))
    weight = dict(zip(model.columns_, model.model_.coef_[0]))
    assert weight["days_since_order"] > 0
    p = model.predict_proba(table(seed=4))
    assert ((p > 0) & (p < 1)).all()
