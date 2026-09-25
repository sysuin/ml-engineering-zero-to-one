"""
Chapter 14's decisions: lists cut by cost and by capacity, calibration
learned out of time and never on the rows it is judged on, revenue at
risk, and the resampling the chapter argues against. Made-up tables
only; no warehouse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table

from foresight.config import rng
from foresight.costs import CostMatrix
from foresight.decide import (COSTS, Calibrated, Isotonic, Platt,
                              by_both, by_capacity, by_cost,
                              calibration_slope, cost_of_errors,
                              expected_cost_of_errors, expected_value,
                              held_back, logit, net_value, oversample,
                              reliability, revenue_at_risk, simulate,
                              smote, undo_weight, value_at_stake)
from foresight.evaluate import backtest
from foresight.models.regularised import RegularisedRisk


def cohorts():
    """Two cohorts of five, with scores and outcomes chosen by hand."""
    return pd.DataFrame({
        "moment": pd.to_datetime(["2024-05-02"] * 5 + ["2024-06-02"] * 5),
        "contract_id": range(1, 11),
        "model": [.9, .5, .5, .1, .2, .05, .3, .12, .13, .6],
        "not_renewed": [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    })


def test_capacity_takes_k_per_cohort_ties_to_lower_id():
    s = cohorts()
    called = by_capacity(s, k=2)
    assert called.sum() == 4
    # 0.5 and 0.5 tie in the first cohort: contract 2 wins over 3.
    assert list(s.contract_id[called]) == [1, 2, 7, 10]


def test_cost_cuts_at_the_break_even():
    costs = CostMatrix(1000, 0.25, 1, 50)          # break-even 0.2
    p = np.array([0.1, 0.2, 0.21, 0.9])
    assert list(by_cost(p, costs)) == [False, False, True, True]


def test_both_is_capacity_and_cost():
    s = cohorts()
    costs = CostMatrix(1000, 0.25, 1, 50)          # break-even 0.2
    both = by_both(s, k=3, costs=costs)
    assert (both == (by_capacity(s, 3) & by_cost(s.model, costs))).all()
    assert both.sum() == 5


def test_net_and_error_costs_add_up():
    s = cohorts()
    y, called = s.not_renewed, by_capacity(s, k=2)
    tp, fp = 3, 1
    fn = int(y.sum()) - tp
    assert net_value(y, called) == pytest.approx(
        tp * (COSTS.save_value - COSTS.call_cost) - fp * COSTS.call_cost)
    err = COSTS.error_costs()
    assert cost_of_errors(y, called) == pytest.approx(
        fp * err["fp"] + fn * err["fn"])
    # Cost of errors is a constant minus net value.
    everyone = int(y.sum()) * err["fn"]
    assert cost_of_errors(y, called) == pytest.approx(
        everyone - net_value(y, called))


def test_expected_cost_is_lowest_at_the_break_even():
    g = rng(3)
    p = g.random(500) * 0.4
    ts = np.linspace(0, 0.4, 81)
    cost = [expected_cost_of_errors(p, p > t) for t in ts]
    best = ts[int(np.argmin(cost))]
    assert abs(best - COSTS.break_even()) <= 0.005


def test_expected_value_matches_hand_arithmetic():
    p = np.array([0.5, 0.1])
    called = np.array([True, False])
    assert expected_value(p, called) == pytest.approx(
        0.5 * COSTS.save_value - COSTS.call_cost)


def test_platt_recovers_a_known_distortion():
    g = rng(4)
    true = 1 / (1 + np.exp(-g.normal(-2.5, 1.2, 20_000)))
    y = (g.random(len(true)) < true).astype(int)
    timid = 1 / (1 + np.exp(-(0.5 * logit(true) - 1.25)))
    a, b = calibration_slope(timid, y)
    assert a == pytest.approx(2.0, abs=0.15)
    fixed = Platt().fit(timid, y)(timid)
    assert np.abs(fixed - true).mean() < 0.01


def test_maps_keep_the_order():
    g = rng(5)
    p = g.random(300)
    y = (g.random(300) < p).astype(int)
    for method in (Platt, Isotonic):
        q = method().fit(p, y)(p)
        order = np.argsort(p)
        assert (np.diff(q[order]) >= -1e-12).all()


def test_reliability_bins_are_equal_and_complete():
    g = rng(6)
    p = g.random(1003)
    y = (g.random(1003) < p).astype(int)
    t = reliability(p, y, bins=10)
    assert t.contracts.sum() == 1003
    assert t.contracts.max() - t.contracts.min() <= 1
    assert (np.diff(t.predicted) > 0).all()
    assert t.leavers.sum() == y.sum()


def test_held_back_is_the_latest_cohorts_and_known_before_them():
    t = table(cohorts=8, per_cohort=50)
    early, held = held_back(t, months=3)
    marks = np.sort(t.moment.unique())
    assert set(held.moment) == set(marks[-3:])
    assert (early.end_date < marks[-3]).all()


def lasso():
    return RegularisedRisk("l1", 0.002)


def test_calibrated_learns_its_map_out_of_time():
    t = table(cohorts=10, per_cohort=100)
    c = Calibrated(lasso).fit(t)
    early, held = held_back(t)
    scores = lasso().fit(early).predict_proba(held)
    ref = Platt().fit(scores, held.not_renewed)
    assert c.map_.a_ == pytest.approx(ref.a_)
    assert c.map_.b_ == pytest.approx(ref.b_)
    # ...and scores with the model refitted on every row.
    assert np.allclose(c.predict_proba(t),
                       ref(lasso().fit(t).predict_proba(t)))


def test_calibrated_runs_through_the_backtest():
    t = table(cohorts=12, per_cohort=80)
    s = backtest(t, "2023-10-01", "2023-12-31",
                 lambda: Calibrated(lasso, "isotonic"))
    assert s.model.between(0, 1).all()
    assert s.moment.nunique() == 3


def test_revenue_at_risk_and_its_simulation():
    p = np.array([0.1, 0.5, 0.2, 0.0])
    v = np.array([100.0, 200.0, 50.0, 1e6])
    groups = np.array(["a", "a", "b", "b"])
    risk = revenue_at_risk(p, v, groups)
    assert risk["a"] == pytest.approx(110)
    assert risk["b"] == pytest.approx(10)
    draws = simulate(p, v, groups, reps=20_000, seed=1)
    assert draws["a"].mean() == pytest.approx(110, rel=0.03)
    assert draws["b"].max() <= 50           # a chance of 0 never leaves


def test_value_at_stake_is_a_quarter_at_the_run_rate():
    rows = pd.DataFrame({"spend_365": [365.0, 0.0]})
    assert list(value_at_stake(rows)) == [90.0, 0.0]


def test_oversample_balances_and_keeps_every_real_row():
    t = table(cohorts=4, per_cohort=100)
    out = oversample(t, rng(1))
    assert (out.not_renewed == 1).sum() == (out.not_renewed == 0).sum()
    assert out.contract_id.isin(t.contract_id).all()
    assert out.iloc[:len(t)].equals(t)


def test_smote_invents_leavers_between_real_ones():
    t = table(cohorts=4, per_cohort=100)
    out = smote(t, rng(1))
    new = out.iloc[len(t):]
    assert (new.not_renewed == 1).all()
    assert (out.not_renewed == 1).sum() == (out.not_renewed == 0).sum()
    real = t[t.not_renewed == 1]
    for c in ("orders_90d", "spend_365", "tenure_days"):
        assert new[c].min() >= real[c].min() - 1e-9
        assert new[c].max() <= real[c].max() + 1e-9
    # Legacy leavers have no discount; some invented ones keep the gap.
    assert new.discount_pct.isna().any()
    assert new.legacy_terms.isin([0, 1]).all()


def test_undo_weight_inverts_a_weighting():
    p = np.array([0.01, 0.1, 0.5, 0.9])
    odds = p / (1 - p) * 14
    weighted = odds / (1 + odds)
    assert np.allclose(undo_weight(weighted, 14), p)
