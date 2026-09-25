"""
Chapter 27's MCP server: three tools, each read-only, each with a
schema that refuses a bad argument before any of Foresight's code
runs; the region's share of the list of record, key accounts apart and
held-out accounts marked; Chapter 3's rule when no list was made; the
API's answer and refusals for one contract; the forecast by category;
nothing written and no path given away. A made-up warehouse and scores
database in a temporary folder; no model, no network.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from foresight import mcp_server as mcp
from foresight.serve import store
from foresight.serve.api import NotYet

pytestmark = pytest.mark.service
MARK = "2025-12-31"


@pytest.fixture
def installation(tmp_path):
    """A warehouse of six contracts in two regions, and a finished
    list for their cohort: four calls, one of them held out, and a key
    account listed apart."""
    wh = tmp_path / "warehouse.db"
    with closing(sqlite3.connect(wh)) as con, con:
        con.executescript("""
            CREATE TABLE regions (region_id INTEGER, name TEXT);
            CREATE TABLE accounts (account_id INTEGER, name TEXT,
                region_id INTEGER, is_key_account INTEGER);
            CREATE TABLE account_history (account_id INTEGER,
                valid_from TEXT, account_manager TEXT);
            CREATE TABLE contracts (contract_id INTEGER,
                account_id INTEGER, end_date TEXT, outcome TEXT);
            INSERT INTO regions VALUES (3, 'Midwest'), (4, 'West');""")
        for i, (region, key) in enumerate(
                [(3, 0), (3, 0), (3, 1), (4, 0), (4, 0), (3, 0)], 1):
            con.execute("INSERT INTO accounts VALUES (?, ?, ?, ?)",
                        (i, f"Account {i}", region, key))
            con.execute("INSERT INTO account_history VALUES"
                        " (?, '2024-01-01', ?)", (i, f"Manager {i}"))
            con.execute("INSERT INTO contracts VALUES (?, ?,"
                        " '2026-03-31', NULL)", (100 + i, i))
    root = tmp_path / "foresight"
    with closing(store.connect(store.path(root))) as con, con:
        con.execute("INSERT INTO runs (run, job, mark, run_on, source,"
                    " status) VALUES (1, 'renewal', ?, ?, 'renewal v1',"
                    " 'finished')", (MARK, MARK))
        rows = [  # contract, account, key, rank, chance, listed, arm
            (101, 1, 0, 2, 0.41234, 1, "held out"),
            (102, 2, 0, 1, 0.61, 1, "called"),
            (103, 3, 1, None, None, 0, None),
            (104, 4, 0, 3, 0.2, 1, "called"),
            (105, 5, 0, 4, 0.15, 1, "held out"),
            (106, 6, 0, 5, 0.05, 0, None)]
        for c, a, key, rank, p, listed, arm in rows:
            con.execute(
                "INSERT INTO scores VALUES (1, ?, ?, ?, ?, ?, ?, ?, 0,"
                " ?, ?, '[]')", (MARK, c, a, key, rank, p, listed, arm,
                                 json.dumps([f"reason for {c}"])))
    return root, wh


def call(server, name, **arguments):
    (reply,) = mcp.exchange(server, [mcp.frame(
        1, "tools/call", name=name, arguments=arguments)])
    return reply["result"].get("isError", False), mcp.content(reply)


def digest(*files) -> str:
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)
                          ).hexdigest()


def test_three_tools_each_read_only_with_strict_arguments(
        installation):
    root, wh = installation
    (reply,) = mcp.exchange(mcp.build(root, wh, on=MARK),
                            [mcp.frame(1, "tools/list")])
    tools = {t["name"]: t for t in reply["result"]["tools"]}
    assert set(tools) == {"renewal_risk", "why", "demand_forecast"}
    for t in tools.values():
        assert t["annotations"]["readOnlyHint"] is True
        assert t["annotations"]["destructiveHint"] is False
    region = tools["renewal_risk"]["inputSchema"]["properties"]["region"]
    assert len(region["enum"]) == 5
    assert tools["renewal_risk"]["inputSchema"]["required"] == ["region"]
    cid = tools["why"]["inputSchema"]["properties"]["contract_id"]
    assert cid["type"] == "integer" and cid["exclusiveMinimum"] == 0


def test_the_region_comes_from_the_list_of_record(installation):
    root, wh = installation
    error, got = call(mcp.build(root, wh, on=MARK), "renewal_risk",
                      region="Midwest")
    assert not error
    (one,) = got["lists"]
    assert one["answered_by"] == "list" and one["model"] == "renewal v1"
    assert (one["contracts"], one["calls"], one["of"]) == (4, 2, 4)
    assert [c["contract_id"] for c in one["at_risk"]] == [102, 101]
    assert one["at_risk"][1]["chance"] == 0.412
    assert one["at_risk"][1]["call"] == "held out: do not call"
    assert one["at_risk"][0]["reasons"] == ["reason for 102"]
    assert one["at_risk"][0]["manager"] == "Manager 2"
    (key,) = one["key_accounts"]
    assert key["contract_id"] == 103 and "chance" not in key


def test_a_month_with_no_list_gets_the_rule(installation, monkeypatch):
    root, wh = installation

    def at_mark(mark, warehouse):
        return pd.DataFrame({
            "contract_id": [101, 102, 103, 104],
            "account_id": [1, 2, 3, 4],
            "moment": pd.Timestamp(mark),
            "days_since_order": pd.array([30, 200, 5, None],
                                         dtype="Int64")})
    monkeypatch.setattr(mcp.features, "at_mark", at_mark)
    listless = root.parent / "no-list-yet"
    error, got = call(mcp.build(listless, wh, on="2026-01-15"),
                      "renewal_risk", region="Midwest", month="2025-12")
    assert not error
    (one,) = got["lists"]
    assert one["answered_by"] == "rule"
    assert [c["contract_id"] for c in one["at_risk"]] == [102, 101]
    assert all(c["chance"] is None for c in one["at_risk"])
    assert one["at_risk"][0]["reasons"] == ["Last order 200 days ago"]


def test_bad_arguments_are_refused_before_foresight_runs(
        installation, monkeypatch):
    root, wh = installation
    ran = []
    monkeypatch.setattr(mcp.Foresight, "cohort",
                        lambda self, *a: ran.append(a))
    server = mcp.build(root, wh, on=MARK)
    for name, arguments in [
            ("renewal_risk", {"region": "Mid-west"}),
            ("renewal_risk", {"region": "Midwest", "month": "2025-13"}),
            ("renewal_risk", {"region": "Midwest", "month": "Dec"}),
            ("why", {"contract_id": 0}),
            ("demand_forecast", {"category": "Mops",
                                 "region": "West"})]:
        error, text = call(server, name, **arguments)
        assert error and "validation error" in text
    assert ran == []


def test_a_month_not_yet_reached_is_refused(installation):
    root, wh = installation
    error, text = call(mcp.build(root, wh, on=MARK), "renewal_risk",
                       region="West", month="2026-02")
    assert error and "no list for 2026-02 yet" in text


class Stub:
    """Service's score(), as the API calls it."""

    def score(self, contract_id, clock):
        if contract_id == 999:
            raise NotYet("contract 999 is not at its mark until"
                         " 2026-03-02")
        return {"contract_id": contract_id, "account": "Account 1",
                "mark": pd.Timestamp(MARK).date(),
                "answered_by": "list", "chance": 0.4123, "rank": 2,
                "reasons": ["reason"], "model": "renewal v1",
                "note": "on the list, held out this month: no call"}


def test_why_is_the_apis_answer_and_its_refusal(installation):
    root, wh = installation
    server = mcp.build(root, wh, on=MARK, service=Stub())
    error, got = call(server, "why", contract_id=101)
    assert not error and got["mark"] == MARK and got["rank"] == 2
    error, text = call(server, "why", contract_id=999)
    assert error and text.endswith("not at its mark until 2026-03-02")


def test_the_forecast_adds_products_but_not_their_ranges(
        installation):
    root, wh = installation
    server = mcp.build(root, wh, on=MARK)
    error, text = call(server, "demand_forecast", category="Cleaning",
                       region="Midwest")
    assert error and "no forecast has been made" in text
    with closing(store.connect(store.path(root))) as con, con:
        con.execute("INSERT INTO runs (run, job, mark, run_on, status)"
                    " VALUES (2, 'forecast', '2025-12-01', ?,"
                    " 'finished')", (MARK,))
        for sku, units in [("MRD-CLE-001", 100.4), ("MRD-CLE-002", 50)]:
            con.execute("INSERT INTO forecasts VALUES (2, '2026-01', 1,"
                        " 'Cleaning', ?, 'Midwest', ?, ?, ?)",
                        (sku, units, units - 10, units + 10))
    error, got = call(server, "demand_forecast", category="Cleaning",
                      region="Midwest")
    assert not error and got["from"] == "2025-12"
    assert got["months"] == [{"month": "2026-01", "units": 150}]
    assert got["products"][0]["months"][0] == {
        "month": "2026-01", "units": 100, "lo": 90, "hi": 110}


def test_nothing_is_written_and_no_path_is_given(installation):
    root, wh = installation
    before = digest(store.path(root), wh)
    server = mcp.build(root, wh, on=MARK, service=Stub())
    replies = mcp.exchange(server, [
        mcp.frame(1, "tools/call", name="renewal_risk",
                  arguments={"region": "West"}),
        mcp.frame(2, "tools/call", name="why",
                  arguments={"contract_id": 104}),
        mcp.frame(3, "tools/call", name="demand_forecast",
                  arguments={"category": "Safety", "region": "West"})])
    assert digest(store.path(root), wh) == before
    said = json.dumps(replies)
    assert str(root.parent) not in said
    assert not (store.path(root).parent / "batch.lock").exists()
