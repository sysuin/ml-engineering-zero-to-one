"""
Chapter 9's regularised renewal model: strength 0 is v0.3's model, the
strength means the same leash at any number of rows, the lasso reaches
exact zeros and ridge does not, and the tuning loop reports every
setting it tried. Made-up tables only: no warehouse needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.config import rng
from foresight.evaluate import backtest
from foresight.models.logistic import RenewalRisk
from foresight.models.regularised import (RegularisedRisk, choose,
                                          head_to_head, tune)


def table(cohorts=8, per_cohort=120, seed=1) -> pd.DataFrame:
    """Chapter 4's columns, monthly cohorts, and a label that a long
    gap since the last order makes more likely; region is noise."""
    g = rng(seed)
    ends = pd.date_range("2023-01-31", periods=cohorts, freq="ME")
    n = cohorts * per_cohort
    end = np.repeat(ends, per_cohort)
    gap = g.integers(1, 200, n).astype(float)
    z = -3.0 + 0.02 * gap + g.normal(0, 0.5, n)
    legacy = (g.random(n) < 0.2).astype(int)
    return pd.DataFrame({
        "contract_id": np.arange(n) + 1,
        "account_id": np.arange(n) + 1000,
        "moment": end - pd.Timedelta(days=90),
        "end_date": end,
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
        "not_renewed": (g.random(n) < 1 / (1 + np.exp(-z))).astype(int),
    })


def test_strength_zero_is_v03():
    t = table()
    a = RegularisedRisk("l2", 0.0).fit(t).predict_proba(t)
    b = RenewalRisk().fit(t).predict_proba(t)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("penalty", ["l2", "l1"])
def test_strength_is_per_row(penalty):
    """Two copies of every row: the same data twice, the same weights."""
    t = table()
    once = RegularisedRisk(penalty, 0.005).fit(t).weights()
    twice = RegularisedRisk(penalty, 0.005).fit(
        pd.concat([t, t], ignore_index=True)).weights()
    assert np.allclose(once, twice, atol=1e-4)


def test_ridge_shrinks_and_lasso_zeroes():
    t = table()
    sizes = [np.abs(RegularisedRisk("l2", s).fit(t).weights()).sum()
             for s in (0.0, 0.001, 0.01, 0.1)]
    assert all(a > b for a, b in zip(sizes, sizes[1:]))
    ridge = RegularisedRisk("l2", 0.01).fit(t).weights()
    lasso = RegularisedRisk("l1", 0.01).fit(t).weights()
    assert (ridge != 0).all()
    assert (lasso == 0).sum() > 0
    assert lasso["days_since_order"] != 0     # the real signal stays
    gone = RegularisedRisk("l1", 1.0).fit(t).weights()
    assert (gone == 0).all()


def test_drop_removes_columns():
    t = table()
    m = RegularisedRisk("l2", 0.001, drop=("region=",)).fit(t)
    assert not any(c.startswith("region=") for c in m.columns_)
    assert len(m.columns_) == 12
    assert m.predict_proba(t).shape == (len(t),)


def test_tune_reports_every_setting_and_choose_picks_lowest():
    t = table()
    r = tune(t, penalties=("l2", "l1"), strengths=(0.0, 0.01),
             first="2023-05-01", last="2023-08-31")
    assert len(r) == 4
    assert set(r.columns) == {"penalty", "strength", "log loss", "auc",
                              "hits"}
    best = choose(r)
    assert r["log loss"].min() == pytest.approx(
        r[(r.penalty == best["penalty"])
          & (r.strength == best["strength"])]["log loss"].iloc[0])


def test_head_to_head_is_paired():
    t = table()
    a = backtest(t, "2023-05-01", "2023-08-31")
    b = backtest(t, "2023-05-01", "2023-08-31",
                 lambda: RegularisedRisk("l1", 0.01))
    both = head_to_head(b, a)
    assert (both.rule.to_numpy() == a.model.to_numpy()).all()
    with pytest.raises(AssertionError):
        head_to_head(b, a.iloc[::-1].reset_index(drop=True))
