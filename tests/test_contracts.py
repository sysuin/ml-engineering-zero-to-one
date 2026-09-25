"""
Chapter 23's data contracts. The checker, clause by clause, on a feed
of a few rows built here: a clean feed keeps its contract, and each
kind of breach is found and named. Then, when the dataset is there,
every feed of the real warehouse against its contract.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from foresight import contracts
from foresight.contracts import (FEEDS, Column, Contract, Rule,
                                 breaches, describe)

pytestmark = pytest.mark.unit
ON = "2025-01-10"

FEED = Contract(
    table="feed", owner="a team", version="1.0",
    columns=(Column("id", "integer", low=1),
             Column("day", "text", form="[0-9][0-9][0-9][0-9]-"
                    "[0-9][0-9]-[0-9][0-9]"),
             Column("kind", "text", values=("a", "b")),
             Column("amount", "real", low=0, high=100),
             Column("note", "text", required=False),
             Column("ref", "integer", refers="refs.ref")),
    key=("id",), when="day", fresh_days=1, daily_rows=2,
    rules=(Rule("no amount of 99", "SELECT COUNT(*) FROM {rows}"
                " WHERE amount = 99"),))


def clean() -> list[tuple]:
    """Three rows a day for the 28 days before ON."""
    days = pd.date_range("2024-12-13", "2025-01-09")
    return [(i * 3 + j + 1, f"{d:%Y-%m-%d}", "ab"[j % 2], 10.0 * j,
             None, 1) for i, d in enumerate(days) for j in range(3)]


def feed(rows) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE refs (ref INTEGER)")
    con.execute("INSERT INTO refs VALUES (1)")
    con.execute("CREATE TABLE feed (id, day, kind, amount, note, ref)")
    con.executemany("INSERT INTO feed VALUES (?,?,?,?,?,?)", rows)
    return con


def clauses_broken(rows) -> list[str]:
    return [b.clause for b in breaches(feed(rows), FEED, ON)]


def test_a_clean_feed_keeps_its_contract():
    assert clauses_broken(clean()) == []


@pytest.mark.parametrize("change, clause", [
    ({"id": "7"}, "id is integer"),
    ({"kind": None}, "kind is never empty"),
    ({"kind": "c"}, "kind is one of 2"),
    ({"amount": -1.0}, "amount >= 0"),
    ({"amount": 101.0}, "amount <= 100"),
    ({"day": "2025/01/05"}, "day looks like 9999-99-99"),
    ({"ref": 2}, "ref is in refs.ref"),
    ({"amount": 99.0}, "no amount of 99"),
])
def test_each_kind_of_breach_is_named(change, clause):
    rows = clean()
    names = ["id", "day", "kind", "amount", "note", "ref"]
    first = dict(zip(names, rows[0])) | change
    rows[0] = tuple(first[n] for n in names)
    assert clause in clauses_broken(rows)


def test_a_date_that_will_not_parse_is_kept_and_named():
    rows = clean()
    rows[-1] = (rows[-1][0], "09/01/2025", *rows[-1][2:])
    broken = breaches(feed(rows), FEED, ON)
    assert any(b.clause.startswith("day looks like")
               and b.example == str(rows[-1][0]) for b in broken)


def test_a_repeated_key_is_a_breach():
    rows = clean()
    rows.append(rows[0])
    assert "one row per id" in clauses_broken(rows)


def test_a_stale_feed_is_a_breach():
    rows = [r for r in clean() if r[1] < "2025-01-08"]
    assert any(c.startswith("rows to within 1 day")
               for c in clauses_broken(rows))


def test_a_short_day_is_a_breach_and_names_the_day():
    rows = [r for r in clean() if not (r[1] == "2025-01-02"
                                       and r[0] % 3 != 0)]
    found = [b for b in breaches(feed(rows), FEED, ON)
             if b.clause.startswith("at least 2 rows a day")]
    assert found and found[0].example == "2025-01-02: 1"


def test_rows_after_the_morning_are_not_judged():
    rows = clean() + [(999, "2025-01-11", "zzz", -5.0, None, 7)]
    assert clauses_broken(rows) == []


def test_a_missing_column_is_a_breach():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE refs (ref INTEGER)")
    con.execute("CREATE TABLE feed (id, day, kind, amount, note)")
    assert "has a column ref" in [b.clause for b in
                                  breaches(con, FEED, None)]


def test_every_contract_fits_the_page():
    for c in (*FEEDS, contracts.ANSWERS):
        assert max(map(len, describe(c).splitlines())) <= 68


# ------------------------------------------------ the real feeds
@pytest.mark.data
@pytest.mark.parametrize("contract", FEEDS, ids=lambda c: c.table)
def test_the_warehouse_keeps_every_feeds_contract(ml_warehouse,
                                                  contract):
    with contracts.read_only(ml_warehouse) as con:
        assert breaches(con, contract, "2025-12-31") == []
