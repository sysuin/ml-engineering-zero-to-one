"""
Chapter 16: explanations that add up, reasons in the right order,
importance that is zero for a column the model ignores, slices that
count correctly, a list page that fits the page, and a model card that
will not be written with a number missing. Made-up tables only; no
warehouse.
"""
from __future__ import annotations

import string

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table

from foresight.card import TEMPLATE, TEST, render
from foresight.config import rng
from foresight.decide import Calibrated
from foresight.explain import (COLUMNS, contributions, facts, log_odds,
                               partial_dependence, permutation_importance,
                               reasons, say, shuffled, unfamiliar)
from foresight.models.boosting import RenewalBooster
from foresight.models.regularised import RegularisedRisk
from foresight.score import page
from foresight.slices import label
from foresight.slices import table as slice_table


def lasso():
    return RegularisedRisk("l1", 0.002)


# ------------------------------------------------ contributions
def test_linear_parts_add_up_to_the_log_odds():
    t = table()
    m = lasso().fit(t)
    base, parts = contributions(m, t)
    assert set(parts.columns) == set(COLUMNS)
    total = base + parts.sum(axis=1).to_numpy()
    assert np.allclose(total, log_odds(m.predict_proba(t)))
    # The base is the average training contract: the intercept.
    assert np.allclose(base, m.model_.intercept_[0])


def test_linear_parts_are_weight_times_standardised_value():
    t = table()
    m = lasso().fit(t)
    _, parts = contributions(m, t.head(5))
    z = m.scaler_.transform(m.columns(t.head(5)).to_numpy(float))
    w = dict(zip(m.columns_, m.model_.coef_[0]))
    col = m.columns_.index("days_since_order")
    assert np.allclose(parts.days_since_order,
                       z[:, col] * w["days_since_order"])
    seg = [i for i, c in enumerate(m.columns_) if c.startswith("segment")]
    assert np.allclose(parts.segment,
                       (z[:, seg] * m.model_.coef_[0][seg]).sum(axis=1))


def test_linear_parts_average_zero_on_the_training_rows():
    t = table()
    _, parts = contributions(lasso().fit(t), t)
    assert np.allclose(parts.mean(), 0, atol=1e-9)


def test_tree_parts_add_up_to_the_log_odds():
    t = table(cohorts=10)
    m = RenewalBooster(most=200, patience=20).fit(t)
    base, parts = contributions(m, t)
    total = base + parts.sum(axis=1).to_numpy()
    assert np.allclose(total, log_odds(m.predict_proba(t)), atol=1e-9)


def test_calibrated_parts_are_the_inner_parts_times_the_slope():
    t = table(cohorts=10)
    m = Calibrated(lasso).fit(t)
    base, parts = contributions(m, t)
    _, inner = contributions(m.model_, t)
    assert np.allclose(parts, inner * m.map_.a_)
    total = base + parts.sum(axis=1).to_numpy()
    assert np.allclose(total, log_odds(m.predict_proba(t)))


def test_an_isotonic_map_has_no_parts():
    t = table(cohorts=10)
    m = Calibrated(lasso, "isotonic").fit(t)
    with pytest.raises(TypeError):
        contributions(m, t)


# ------------------------------------------------ reasons
def rows_and_parts():
    t = table().head(2).reset_index(drop=True)
    t.loc[0, ["orders_90d", "orders_prev_90d"]] = [1, 6]
    parts = pd.DataFrame(0.0, index=t.index, columns=COLUMNS)
    parts.loc[0, ["days_since_order", "segment", "orders_90d",
                  "orders_prev_90d", "tenure_days"]] = [0.9, 0.3, 0.5,
                                                        -0.1, 0.04]
    parts.loc[1, ["discount_pct", "region"]] = [-0.4, 0.2]
    return t, parts


def test_reasons_are_the_largest_rises_first():
    t, parts = rows_and_parts()
    why = reasons(parts, t, facts(table()))
    assert len(why[0]) == 3
    assert why[0][0].startswith("Last order")
    # The two order counts are one reason: 0.5 - 0.1 = 0.4 > 0.3.
    assert why[0][1] == "Orders fell from 6 to 1, quarter on quarter"
    assert why[0][2].startswith(t.segment[0])


def test_small_and_falling_parts_are_not_reasons():
    t, parts = rows_and_parts()
    why = reasons(parts, t, facts(table()), smallest=0.05)
    assert not any("customer" in w for w in why[0])     # 0.04
    assert len(why[1]) == 1 and why[1][0].startswith("the ")


def test_say_handles_gaps():
    t = table().head(1)
    r = t.assign(days_since_order=pd.NA,
                 discount_pct=pd.NA).iloc[0]
    f = facts(table())
    assert say("days_since_order", r, f) == "No order on record"
    assert say("discount_pct", r, f) == "No discount on record"


def test_unfamiliar_flags_the_far_tails_only():
    train = pd.DataFrame({"x": np.arange(100.0)})
    rows = pd.DataFrame({"x": [50.0, 95.0, 85.0, 500.0, -3.0, np.nan]})
    odd = unfamiliar(train, rows, ["x"], rare=10)
    assert list(odd.x) == [False, True, False, True, True, False]


# ------------------------------------------------ importance
def test_shuffles_stay_inside_each_cohort_and_move_together():
    t = table()
    s = shuffled(t, ("orders_90d", "orders_prev_90d"), rng(1))
    for (_, a), (_, b) in zip(t.groupby("moment"), s.groupby("moment")):
        assert sorted(a.orders_90d) == sorted(b.orders_90d)
    pairs = set(zip(t.orders_90d, t.orders_prev_90d))
    assert set(zip(s.orders_90d, s.orders_prev_90d)) <= pairs
    assert (s.days_since_order == t.days_since_order).all()


def test_importance_is_zero_for_a_column_the_score_ignores():
    t = table()

    def score(rows):
        return rows.days_since_order.astype(float).to_numpy()

    imp = permutation_importance(t, score, ["tickets_90d",
                                            "days_since_order"],
                                 repeats=5, k=10)
    assert imp.loc["tickets_90d"].abs().max() == 0
    assert imp.loc["days_since_order", "auc"] > 0.05


def test_importance_repeats_with_its_seed():
    t = table()

    def score(rows):
        return rows.days_since_order.astype(float).to_numpy()

    a = permutation_importance(t, score, ["days_since_order"], 5)
    b = permutation_importance(t, score, ["days_since_order"], 5)
    pd.testing.assert_frame_equal(a, b)


def test_partial_dependence_sets_the_column_everywhere():
    class Echo:
        def predict_proba(self, rows):
            return rows.tickets_90d.astype(float).to_numpy() / 10

    pd_ = partial_dependence(Echo(), table(), "tickets_90d", [0, 1, 5])
    assert np.allclose(pd_, [0.0, 0.1, 0.5])


# ------------------------------------------------ slices
def scored_by_hand():
    return pd.DataFrame({
        "moment": pd.to_datetime(["2024-05-02"] * 6),
        "contract_id": range(6), "account_id": range(6),
        "region": ["A", "A", "A", "B", "B", "B"],
        "not_renewed": [1, 1, 0, 1, 0, 0],
        "model": [0.5, 0.1, 0.2, 0.3, 0.3, 0.1]})


def test_slice_rates_by_hand():
    s = scored_by_hand()
    called = [True, False, True, False, True, False]
    t = slice_table(s, "region", called, reps=50)
    assert t.loc["A", "miss"] == 0.5            # one of two leavers
    assert t.loc["A", "fpr"] == 1.0             # its one renewer
    assert t.loc["B", "miss"] == 1.0
    assert t.loc["B", "fpr"] == 0.5
    assert np.isclose(t.loc["A", "gap"], 0.8 / 3 - 2 / 3)
    assert t.loc["A", "gap_lo"] <= t.loc["A", "gap"] <= t.loc["A",
                                                              "gap_hi"]


def test_slices_cut_size_on_the_training_rows():
    train = pd.DataFrame({"spend_365": np.arange(1.0, 101.0)})
    s = scored_by_hand().assign(spend_365=[1, 30, 60, 99, 50, 26],
                                tenure_days=[100, 400, 1200, 2500,
                                             3000, 10])
    out = label(s, train, keys={5})
    assert list(out["size"]) == ["smallest", "second", "third",
                                 "largest", "second", "second"]
    assert list(out.tenure)[:4] == ["under 1 year", "1-3 years",
                                    "3-6 years", "over 6 years"]
    assert list(out.population) == ["long tail"] * 5 + ["key accounts"]


# ------------------------------------------------ the list and the card
def test_the_list_page_fits_the_page():
    listed = pd.DataFrame({
        "rank": [1, 2], "contract_id": [10474, 10415],
        "name": ["A very long account name indeed Ltd", "Short"],
        "account_manager": ["Somebody Long-Named", "Kit"],
        "chance": [0.82, 0.11], "below": [False, True],
        "reasons": [["Last order 243 days ago (typical: 13)"], []],
        "unfamiliar": [[], ["orders_90d"]]})
    keys = pd.DataFrame({"contract_id": [1], "name": ["Key"],
                         "account_manager": ["X"], "orders_90d": [611],
                         "orders_prev_90d": [678], "tickets_90d": [11]})
    result = {"mark": pd.Timestamp("2024-10-02"), "contracts": 364,
              "trained_on": 6065, "as_of": "2024-10-02",
              "list": listed, "keys": keys}
    text = page(result)
    assert max(len(line) for line in text.splitlines()) <= 68
    assert "below" in text and "rarely seen: orders_90d" in text


def fields(template):
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def test_the_card_will_not_render_with_a_number_missing():
    numbers = dict.fromkeys(fields(TEMPLATE), "1")
    assert "{" not in render(numbers)
    numbers.pop("hits")
    with pytest.raises(KeyError):
        render(numbers)


def test_the_test_year_section_is_added_only_when_given():
    numbers = dict.fromkeys(fields(TEMPLATE), "1")
    assert "Test year" not in render(numbers)
    test = dict.fromkeys(fields(TEST), "2")
    assert "Test year, read once" in render(numbers, test)
