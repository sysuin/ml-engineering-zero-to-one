"""
The training-table builder from Chapter 4.

Most tests run against a warehouse of a few rows, built here, where every
feature can be worked out by hand: an order on the day of the moment, a
segment that changes that day, a legacy id, a contract with no outcome.
The last test builds a slice of the real table and runs its checks.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from foresight import config
from foresight.data.build_table import (TableCheckError, build, check,
                                        events, expected_rows, legacy_ids,
                                        tidy, write)

SCHEMA = """
CREATE TABLE regions (region_id INTEGER, name TEXT, depot TEXT);
CREATE TABLE accounts (account_id INTEGER, name TEXT, region_id INTEGER,
    segment TEXT, account_manager TEXT, since TEXT, postcode TEXT,
    is_key_account INTEGER, crm_source TEXT, closed_on TEXT);
CREATE TABLE account_history (account_id INTEGER, valid_from TEXT,
    valid_to TEXT, segment TEXT, account_manager TEXT);
CREATE TABLE contracts (contract_id INTEGER, account_id INTEGER,
    start_date TEXT, end_date TEXT, term_months INTEGER,
    legacy_terms INTEGER, discount_pct INTEGER, outcome TEXT,
    notice_date TEXT, cancellation_reason TEXT);
CREATE TABLE orders (order_id INTEGER, account_id INTEGER,
    order_date TEXT, channel TEXT);
CREATE TABLE order_lines (order_id INTEGER, line_no INTEGER, sku TEXT,
    qty INTEGER, unit_price REAL, unit_cost REAL, discount_pct INTEGER);
CREATE TABLE tickets (ticket_id TEXT, account_id INTEGER, opened_at TEXT,
    channel TEXT, sku TEXT, language TEXT, body TEXT, desk TEXT,
    category TEXT, priority TEXT);
"""

ACCOUNTS = [
    (1, "Acme Cleaning", 1, "Mid-market", "A", "2020-01-01", "12345", 0,
     "Meridian CRM", "2024-03-31"),
    (2, "Birch Dental", 1, "Public sector", "B", "2021-06-01", "54321", 0,
     "Meridian CRM", None),
    (90001, "ACME  CLEANING (old)", 1, "Small business", "A",
     "2020-01-01", "12345", 0, "Legacy CRM", None),
    # Same postcode as Acme, a different start date: not the same firm.
    (90002, "Acme Cleaning", 1, "Small business", "C", "2019-05-01",
     "12345", 0, "Legacy CRM", None),
]
HISTORY = [
    (1, "2022-01-01", "2023-12-31", "Small business", "A"),
    (1, "2024-01-01", None, "Mid-market", "A"),   # changes on the moment
    (2, "2022-01-01", None, "Public sector", "B"),
]
CONTRACTS = [   # contract 10's moment is 2024-01-01
    (10, 1, "2023-04-01", "2024-03-31", 12, 0, 5, "not_renewed", None,
     None),
    (11, 2, "2023-04-01", "2024-03-31", 12, 1, None, "renewed", None, None),
    (12, 2, "2024-04-01", "2025-03-31", 12, 1, None, None, None, None),
]
ORDERS = [      # (order_id, account_id, day, value)
    (1, 90001, "2022-12-01", 70.0),  # 396 days before: outside the year
    (2, 90001, "2023-09-15", 50.0),   # 108 days before: the previous 90
    (3, 90001, "2023-12-01", 100.0),  # 31 days before: the last order
    (4, 1, "2024-01-01", 999.0),      # on the day of the moment
    (5, 1, "2024-02-01", 999.0),      # after it
    (6, 2, "2023-12-31", 20.0),
]
TICKETS = [
    ("T1", 90001, "2023-12-20 10:00:00"),
    ("T2", 1, "2024-01-01 09:00:00"),     # on the day of the moment
]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("warehouse") / "tiny.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("INSERT INTO regions VALUES (1, 'Midwest', 'Columbus')")
    con.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?)",
                    ACCOUNTS)
    con.executemany("INSERT INTO account_history VALUES (?,?,?,?,?)",
                    HISTORY)
    con.executemany("INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?)",
                    CONTRACTS)
    for oid, acct, day, value in ORDERS:
        con.execute("INSERT INTO orders VALUES (?,?,?,'Web')",
                    (oid, acct, day))
        con.execute("INSERT INTO order_lines VALUES (?,1,'X',1,?,1.0,0)",
                    (oid, value))
    con.executemany("INSERT INTO tickets (ticket_id, account_id, opened_at)"
                    " VALUES (?,?,?)", TICKETS)
    con.commit()
    con.close()
    return path


@pytest.fixture(scope="module")
def table(tiny):
    return build("2024-01-01", "2025-12-31", warehouse=tiny)


def row(table, contract_id):
    return table.set_index("contract_id").loc[contract_id]


def test_tidy_agrees_on_the_names_the_migration_produced():
    for name in ("ACME CLEANING", "Acme  Cleaning", "Acme Cleaning Inc",
                 "Acme Cleaning (old)"):
        assert tidy(name) == "acme cleaning"


def test_legacy_ids_need_postcode_start_date_and_name(tiny):
    with sqlite3.connect(tiny) as con:
        assert legacy_ids(con) == {90001: 1}


def test_events_refuse_history_left_under_a_legacy_id(tiny):
    with sqlite3.connect(tiny) as con, pytest.raises(TableCheckError):
        events(con, {})               # no mapping: 90001's orders strand


def test_features_read_only_what_came_before_the_moment(table):
    r = row(table, 10)
    assert r.account_id == 1  # the legacy id is resolved
    assert r.days_since_order == 31           # not the order on the day
    assert r.orders_90d == 1
    assert r.orders_prev_90d == 1
    assert r.spend_365 == 150.0  # 50 + 100: not 70, not 999
    assert r.tickets_90d == 1  # not the ticket on the day
    assert r.tenure_days == 1461


def test_segment_is_the_one_in_force_before_the_moment(table):
    assert row(table, 10).segment == "Small business"


def test_missing_discount_stays_missing(table):
    assert pd.isna(row(table, 11).discount_pct)


def test_unknown_outcomes_are_left_out(tiny, table):
    assert set(table.contract_id) == {10, 11}
    assert build("2024-01-01", "2025-12-31", known_by="2024-03-31",
                 warehouse=tiny).empty


def test_the_table_passes_its_own_checks(tiny, table):
    expected = expected_rows("2024-01-01", "2025-12-31", warehouse=tiny)
    assert check(table, "2024-01-01", "2025-12-31", None, expected) == []


def test_checks_catch_duplicates_legacy_ids_and_late_orders(table):
    bad = pd.concat([table, table.head(1)])
    bad.loc[bad.contract_id == 11, "account_id"] = 90001
    bad["days_since_order"] = bad.days_since_order.astype("Int64") * 0
    problems = " | ".join(check(bad, "2024-01-01", "2025-12-31", None, 2))
    assert "3 rows, expected 2" in problems
    assert "1 contracts appear more than once" in problems
    assert "legacy CRM id" in problems
    assert "on or after the moment" in problems


def test_checks_catch_an_empty_table_and_dates_out_of_range(table):
    assert "the table is empty" in check(table.head(0), "2024-01-01",
                                         "2025-12-31")
    assert any("outside" in p for p in check(table, "2024-06-01",
                                             "2025-12-31"))


def test_write_refuses_a_failing_table_and_writes_nothing(tiny, table,
                                                          tmp_path):
    path = tmp_path / "t.parquet"
    with pytest.raises(TableCheckError):
        write(pd.concat([table, table]), "2024-01-01", "2025-12-31",
              path=path, warehouse=tiny)
    assert not path.exists()


def test_write_saves_the_table_and_its_manifest(tiny, table, tmp_path):
    path = tmp_path / "t.parquet"
    manifest = write(table, "2024-01-01", "2025-12-31", path=path,
                     warehouse=tiny)
    assert pd.read_parquet(path).equals(table)
    saved = json.loads(path.with_suffix(".manifest.json").read_text())
    assert saved == manifest
    assert saved["rows"] == 2 and saved["checks"] == "passed"


def test_a_slice_of_the_real_table_passes_its_checks():
    if not config.ML_WAREHOUSE.exists():
        pytest.skip("the Meridian ML dataset is not generated: run make data")
    first, last = "2023-01-01", "2023-03-31"
    real = build(first, last)
    assert check(real, first, last, None, expected_rows(first, last)) == []
    assert real.account_id.lt(90_000).all()
