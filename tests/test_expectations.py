"""
The expectations from Chapter 5.

A small table is built by hand that obeys every rule: twenty renewals
in the training split, two of them leavers, three on legacy terms, one
key account marked before its record starts. Each test then breaks one
rule the way a real mistake would, and checks that check_table()
refuses the table and names the rule. The last test runs every
expectation on the real table.
"""
from __future__ import annotations

import pandas as pd
import pytest

from foresight import config
from foresight.data.build_table import TABLE
from foresight.data.expectations import (EXPECTATIONS, ExpectationError,
                                         check_table, failures, split_of)


def good_table() -> pd.DataFrame:
    """Twenty rows that break nothing."""
    n = 20
    end = pd.to_datetime(["2023-06-30"] * 10 + ["2024-03-31"] * 10)
    t = pd.DataFrame({
        "contract_id": range(1, n + 1),
        "account_id": range(101, 101 + n),
        "moment": end - pd.Timedelta(days=90),
        "end_date": end,
        "days_since_order": pd.array([5, 40, 120, 400] + [10] * 16,
                                     dtype="Int64"),
        "orders_90d": [3, 1, 0, 0] + [2] * 16,
        "orders_prev_90d": [2] * n,
        "spend_365": [900.0, 500.0, 80.0, 0.0] + [1200.0] * 16,
        "tickets_90d": [0] * n,
        "tenure_days": [274] * n,
        "segment": ["Small business", "Mid-market", "Public sector",
                    "Enterprise"] * 5,
        "region": ["Northeast", "Southeast", "Midwest", "Southwest",
                   "West"] * 4,
        "term_months": [12] * 18 + [24] * 2,
        "legacy_terms": [1, 1, 1] + [0] * 17,
        "discount_pct": pd.array([None] * 3 + [0, 3, 5, 8, 10, 12, 15] * 2
                                 + [5, 5, 5], dtype="Int64"),
        "not_renewed": [1] + [0] * 9 + [1] + [0] * 9,
    })
    # A key account whose mark fell before its record: no history.
    t.loc[0, "moment"] = pd.Timestamp("2022-12-31")
    t.loc[0, "end_date"] = pd.Timestamp("2023-03-31")
    t.loc[0, ["orders_90d", "orders_prev_90d", "spend_365"]] = 0
    t.loc[0, "days_since_order"] = pd.NA
    t.loc[0, "segment"] = None
    return t


def broken_rules(table) -> list[str]:
    with pytest.raises(ExpectationError) as refusal:
        check_table(table)
    return [e.rule for e in EXPECTATIONS if e.rule in str(refusal.value)]


def test_the_hand_built_table_passes():
    assert failures(good_table()) == []
    check_table(good_table())


def test_filling_discount_gaps_with_zero_is_refused():
    t = good_table()
    t["discount_pct"] = t.discount_pct.fillna(0)
    assert broken_rules(t) == [
        "discount_pct is missing exactly when legacy_terms is 1"]


def test_a_discount_on_a_legacy_contract_is_refused():
    t = good_table()
    t.loc[1, "discount_pct"] = 5
    assert "discount_pct is missing exactly when legacy_terms is 1" \
        in broken_rules(t)


def test_a_discount_off_the_price_list_is_refused():
    t = good_table()
    t.loc[5, "discount_pct"] = 6        # the mean, say, rounded
    assert broken_rules(t) == [
        "discount_pct is on the price list, or missing"]


def test_zero_recency_is_refused():
    t = good_table()
    t["days_since_order"] = t.days_since_order.fillna(0)
    rules = broken_rules(t)
    assert "days_since_order runs from 1 to the first order on record" \
        in rules
    assert "spend_365 is 0 exactly when no order in the year" in rules


def test_recency_older_than_the_record_is_refused():
    t = good_table()
    t.loc[4, "days_since_order"] = 2_000
    assert "days_since_order runs from 1 to the first order on record" \
        in broken_rules(t)


def test_spend_without_an_order_is_refused():
    t = good_table()
    t.loc[3, "spend_365"] = 50.0       # last order 400 days ago
    assert broken_rules(t) == [
        "spend_365 is 0 exactly when no order in the year"]


def test_orders_disagreeing_with_recency_are_refused():
    t = good_table()
    t.loc[2, "orders_90d"] = 1         # last order 120 days ago
    assert broken_rules(t) == [
        "orders_90d is 0 exactly when no order in 90 days"]


def test_negative_values_are_refused():
    t = good_table()
    t.loc[6, "spend_365"] = -10.0
    assert "counts and spend are never negative" in broken_rules(t)


def test_unknown_categories_are_refused():
    t = good_table()
    t.loc[7, "segment"] = "Unknown"
    t.loc[8, "region"] = "North"
    t.loc[9, "term_months"] = 18
    rules = broken_rules(t)
    assert "segment is one of four, or missing" in rules
    assert "region is one of five" in rules
    assert "term_months is 12 or 24" in rules


def test_a_missing_segment_after_the_record_starts_is_refused():
    t = good_table()
    t.loc[12, "segment"] = None
    assert broken_rules(t) == [
        "segment is missing only before key accounts' history"]


def test_a_label_rate_far_from_training_is_refused():
    t = good_table()
    t.loc[:5, "not_renewed"] = 1        # 7 leavers in 20
    assert broken_rules(t) == [
        "the label rate in each split is between 3% and 13%"]


def test_split_names():
    ends = pd.Series(pd.to_datetime(["2023-01-31", "2024-07-31",
                                     "2025-12-31", "2022-12-31"]))
    assert split_of(ends).tolist() == ["train", "validation", "test",
                                       pd.NA]


def test_the_real_table_meets_every_expectation():
    if not TABLE.exists() or not config.ML_WAREHOUSE.exists():
        pytest.skip("build the table first: python -m "
                    "foresight.data.build_table")
    assert failures(pd.read_parquet(TABLE)) == []
