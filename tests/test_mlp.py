"""
Chapter 19's network: the layers have the shapes the chapter draws, one
neuron's gradient is Chapter 7's, and RenewalMLP fits, predicts and
repeats itself on a made-up table small enough to train in a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from foresight.config import rng
from foresight.evaluate import auc
from foresight.models.logistic import sigmoid
from foresight.models.mlp import (RenewalMLP, epoch, known_split,
                                  log_loss_of, network)


def table(n_per_cohort=60, cohorts=6, seed=0) -> pd.DataFrame:
    """Chapter 4's columns, made up, with leaving driven by the gap."""
    g = rng(seed)
    n = n_per_cohort * cohorts
    gap = g.integers(1, 200, n).astype(float)
    legacy = (g.random(n) < 0.25).astype(int)
    z = -3 + 0.025 * gap
    moments = pd.date_range("2023-04-02", periods=cohorts, freq="MS")
    return pd.DataFrame({
        "contract_id": np.arange(n) + 1,
        "moment": np.repeat(moments, n_per_cohort),
        "days_since_order": pd.array(gap).astype("Int64"),
        "orders_90d": g.integers(0, 20, n),
        "orders_prev_90d": g.integers(0, 20, n),
        "spend_365": g.lognormal(9, 1, n),
        "tickets_90d": g.integers(0, 3, n),
        "tenure_days": g.integers(100, 4000, n),
        "segment": g.choice(["Small business", "Enterprise"], n),
        "region": g.choice(["Midwest", "Southwest"], n),
        "term_months": g.choice([12, 24], n),
        "legacy_terms": legacy,
        "discount_pct": pd.array(np.where(legacy == 1, np.nan, 5.0),
                                 dtype="Float64").astype("Int64"),
        "not_renewed": (g.random(n) < sigmoid(z)).astype(int),
    })


def test_network_shapes_and_parameter_count():
    torch.manual_seed(0)
    net = network(16, hidden=(32, 32), dropout=0.0)
    assert net(torch.zeros(5, 16)).shape == (5, 1)
    n = sum(p.numel() for p in net.parameters())
    assert n == 16 * 32 + 32 + 32 * 32 + 32 + 32 + 1


def test_one_neuron_has_chapter_7s_gradient():
    """No hidden layer: the network is logistic regression, and
    backpropagation gives X.T (p - y) / n."""
    g = rng(1)
    X, y = g.normal(size=(50, 3)), (g.random(50) < 0.3).astype(float)
    torch.manual_seed(0)
    net = network(3, hidden=()).double()
    loss = torch.nn.BCEWithLogitsLoss()(
        net(torch.tensor(X)).squeeze(1), torch.tensor(y))
    loss.backward()
    w = net[0].weight.detach().numpy()[0]
    b = net[0].bias.item()
    p = sigmoid(X @ w + b)
    np.testing.assert_allclose(net[0].weight.grad.numpy()[0],
                               X.T @ (p - y) / len(y), atol=1e-12)
    assert net[0].bias.grad.item() == pytest.approx((p - y).mean())


def test_an_epoch_lowers_the_loss_on_a_learnable_problem():
    g = rng(2)
    X = torch.tensor(g.normal(size=(400, 2)), dtype=torch.float32)
    y = (X[:, 0] > 0).float()
    torch.manual_seed(0)
    net = network(2, hidden=(8,), dropout=0.0)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    before = log_loss_of(net, X, y)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        epoch(net, opt, X, y, 32, gen)
    assert log_loss_of(net, X, y) < before


def test_fit_predicts_probabilities_and_repeats_itself():
    rows = table()
    kw = {"hidden": (8,), "lr": 0.01, "max_epochs": 30, "patience": 5}
    a, b = RenewalMLP(**kw).fit(rows), RenewalMLP(**kw).fit(rows)
    p = a.predict_proba(rows)
    assert p.shape == (len(rows),)
    assert ((p > 0) & (p < 1)).all()
    np.testing.assert_array_equal(p, b.predict_proba(rows))
    assert 1 <= a.epochs_ <= 30
    assert len(a.history_) <= 30
    assert auc(rows.not_renewed, p) > 0.7       # the gap is learned


def test_the_seed_changes_the_starting_point():
    rows = table()
    a = RenewalMLP(hidden=(8,), max_epochs=3, seed=1).fit(rows)
    b = RenewalMLP(hidden=(8,), max_epochs=3, seed=2).fit(rows)
    assert not np.array_equal(a.predict_proba(rows),
                              b.predict_proba(rows))


def test_known_split_uses_only_outcomes_known_by_the_first_mark():
    ends = pd.to_datetime(["2023-06-30", "2024-04-30", "2024-05-31",
                           "2024-07-31", "2024-12-31", "2025-01-31"])
    t = pd.DataFrame({"end_date": ends,
                      "moment": ends - pd.Timedelta(days=90)})
    train, valid = known_split(t)
    assert list(valid.end_date.dt.month) == [7, 12]
    assert (train.end_date < valid.moment.min()).all()
    assert list(train.end_date.dt.month) == [6, 4]
