"""
Chapter 6's regression target on Chapter 4's tiny warehouse: an order on
the day of the mark belongs to the target, never to the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.data.build_table import build
from foresight.data.spend import (baselines, features, mae, rmse,
                                  spend_table, year_on_record)
from test_build_table import tiny  # noqa: F401  the shared warehouse


@pytest.fixture(scope="module")
def spend(tiny, tmp_path_factory):  # noqa: F811
    path = tmp_path_factory.mktemp("table") / "table.parquet"
    build("2024-01-01", "2025-12-31", warehouse=tiny).to_parquet(path)
    return spend_table("2024-01-01", "2025-12-31", table=path,
                       warehouse=tiny).set_index("contract_id")


def test_spend_before_the_mark_stops_the_day_before(spend):
    assert spend.loc[10, "spend_90d"] == 100.0    # 31 days before only


def test_the_target_starts_on_the_day_of_the_mark(spend):
    assert spend.loc[10, "spend_next_90d"] == 1998.0   # both 999s


def test_an_account_with_nothing_after_the_mark_spent_zero(spend):
    assert spend.loc[11, "spend_next_90d"] == 0.0


def test_features_fill_a_missing_gap_with_a_year():
    rows = pd.DataFrame({c: [1.0] for c in (
        "spend_90d", "spend_365", "orders_90d", "orders_prev_90d",
        "tickets_90d", "tenure_days", "term_months", "legacy_terms")})
    rows["days_since_order"] = pd.array([None], dtype="Int64")
    assert features(rows)[0, 4] == 365.0


def test_a_year_on_record_needs_the_whole_year(tiny):  # noqa: F811
    rows = pd.DataFrame({
        "moment": pd.to_datetime(["2023-12-01", "2023-12-02"]),
        "is_key_account": [0, 0]})
    # The tiny warehouse's first order is on 2022-12-01.
    assert year_on_record(rows, tiny).tolist() == [True, True]
    early = rows.assign(moment=pd.to_datetime(["2023-11-30",
                                               "2023-06-01"]))
    assert year_on_record(early, tiny).tolist() == [False, False]


def test_baselines_and_errors():
    train = pd.DataFrame({"spend_next_90d": [10.0, 30.0]})
    rows = pd.DataFrame({"spend_90d": [5.0], "spend_365": [365.0]})
    guesses = baselines(train, rows)
    assert [g[0] for g in guesses.values()] == [20.0, 5.0, 90.0]
    assert mae([1.0, 3.0], np.array([2.0, 2.0])) == 1.0
    assert rmse([0.0, 0.0], np.array([3.0, 4.0])) == pytest.approx(
        np.sqrt(12.5))
