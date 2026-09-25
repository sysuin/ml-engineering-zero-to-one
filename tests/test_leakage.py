"""
Chapter 13's leakage checks, on made-up data: every check must fail a
column that leaks the way it was written to catch, pass one that does
not, and stop the job only when something fails.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from foresight.checks.leakage import (CEILING_AUC, MARGIN, LeakageError,
                                      Report, adversarial, audit,
                                      ceiling_check, run_checks, screen,
                                      single_auc, truncate)
from foresight.features.registry import Feature
from foresight.features.sources import Sources

MARKS = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]


def rows(n=400, seed=0) -> pd.DataFrame:
    """Contracts on two marks, a tenth of them leavers, with a column
    that is the label, one that is noise and one that is a date filled
    only for leavers."""
    g = np.random.default_rng(seed)
    y = (g.random(n) < 0.1).astype(int)
    y[:2] = [1, 0]
    moment = np.where(np.arange(n) % 2, MARKS[1], MARKS[0])
    end = pd.to_datetime(moment) + pd.Timedelta(days=90)
    return pd.DataFrame({
        "contract_id": np.arange(1, n + 1),
        "account_id": np.arange(1, n + 1),
        "moment": pd.to_datetime(moment), "end_date": end,
        "not_renewed": y, "noise": g.normal(size=n),
        "the_label": y.astype(float),
        "reason": np.where(y == 1, "moved to a competitor", None),
        "notice": pd.to_datetime(np.where(
            y == 1, end - pd.Timedelta(days=60), pd.NaT)),
    })


def sources(t: pd.DataFrame) -> Sources:
    """One order a week for every account, from a year before the
    first mark to two months after the last."""
    days = pd.date_range("2023-01-02", "2024-04-01", freq="7D")
    orders = pd.DataFrame([(a, d, 100.0) for a in t.account_id
                           for d in days],
                          columns=["account_id", "day", "value"])
    tickets = orders[["account_id", "day"]].assign(
        category="Delivery", sku=None, body="late")
    lines = orders.assign(category="Cleaning", supplier="Kestrel Supply")
    accounts = pd.DataFrame({"account_id": t.account_id,
                             "since": pd.Timestamp("2020-01-01"),
                             "is_key_account": 0, "postcode": "000"})
    return Sources(orders, lines, tickets, accounts)


@pytest.fixture
def warehouse(tmp_path):
    """The two tables the audit reads, for the contracts in rows()."""
    t = rows()
    path = tmp_path / "w.db"
    con = sqlite3.connect(path)
    pd.DataFrame({"account_id": t.account_id, "valid_from": "2020-01-01",
                  "segment": "Mid-market"}).to_sql(
        "account_history", con, index=False)
    pd.DataFrame({"contract_id": t.contract_id,
                  "start_date": "2023-04-01"}).to_sql(
        "contracts", con, index=False)
    con.close()
    return path


def before_mark(ctx):
    e = ctx.window("orders", 90)
    return ctx.per_contract(e.groupby("contract_id").size())


def to_the_end(ctx):
    e = ctx.keys.merge(ctx.sources.orders, on="account_id")
    e = e[e.day < e.end_date]
    return ctx.per_contract(e.groupby("contract_id").size())


CLEAN = Feature("orders_before", "t", "", ("orders",), 90, "raw", "",
                before_mark)
LEAKY = Feature("orders_to_end", "t", "", ("orders",), 90, "raw", "",
                to_the_end)
LATE = Feature("manager_now", "t", "", ("accounts.account_manager",),
               0, "raw", "", before_mark)


# ------------------------------------------------ the screen
def test_the_label_itself_fails_the_screen():
    t = rows()
    assert single_auc(t.the_label, t.not_renewed, t.account_id) == 1.0
    out = screen(t, ["the_label", "noise"])
    assert out.loc["the_label", "screen"] == "fail"
    assert out.loc["noise", "screen"] == "pass"


def test_a_category_and_a_date_filled_only_for_leavers_fail():
    out = screen(rows(), ["reason", "notice"])
    assert (out.screen == "fail").all()


def test_review_sits_between_a_coin_and_the_ceiling():
    t = rows(4000, seed=1)
    g = np.random.default_rng(2)
    t["half"] = t.not_renewed + g.normal(0, 0.55, len(t))
    a = single_auc(t.half, t.not_renewed, t.account_id)
    verdict = screen(t, ["half"]).loc["half", "screen"]
    review = 0.5 + 0.9 * (CEILING_AUC - 0.5)
    expected = ("fail" if a >= CEILING_AUC else
                "review" if a >= review else "pass")
    assert verdict == expected


# ------------------------------------------------ the audit
def test_truncate_keeps_only_records_before_the_mark():
    t = rows(10)
    cut = truncate(sources(t), MARKS[0])
    for df in (cut.orders, cut.lines, cut.tickets):
        assert (df.day < MARKS[0]).all()


def test_the_deletion_test_catches_a_window_to_the_end(warehouse):
    t = rows(20)
    out = audit(t, ["orders_before", "orders_to_end"], sources(t),
                warehouse, marks=MARKS, extra=(CLEAN, LEAKY))
    assert out.loc["orders_before", "audit"] == "pass"
    assert out.loc["orders_before", "moved"] == 0
    assert out.loc["orders_to_end", "audit"] == "fail"
    assert out.loc["orders_to_end", "moved"] > 0


def test_a_source_written_late_or_undeclared_fails_untested(warehouse):
    t = rows(20)
    out = audit(t, ["manager_now", "mystery"], sources(t), warehouse,
                marks=MARKS, extra=(LATE,))
    assert out.loc["manager_now", "kind"] == "written late"
    assert out.loc["mystery", "kind"] == "undeclared"
    assert (out.audit == "fail").all()
    assert out.moved.isna().all()


def test_chapter_4s_columns_pass_the_audit(warehouse):
    t = rows(20)
    cols = ["days_since_order", "orders_90d", "tickets_90d", "segment",
            "term_months", "tenure_days", "region"]
    out = audit(t, cols, sources(t), warehouse, marks=MARKS)
    assert (out.audit == "pass").all()
    assert out.loc[["tenure_days", "region"], "moved"].isna().all()


# ------------------------------------------------ adversarial
def test_a_clock_fails_and_noise_passes():
    t = rows(600)
    t["clock"] = np.arange(len(t), dtype=float)
    old, new = t.iloc[:300], t.iloc[300:]
    together, each = adversarial(old, new, ["clock", "noise"])
    assert each.loc["clock", "shift"] == "fail"
    assert each.loc["noise", "shift"] == "pass"
    assert together > 0.9


def test_adversarial_reads_no_label():
    t = rows(600).drop(columns="not_renewed")
    t["clock"] = np.arange(len(t), dtype=float)
    _, each = adversarial(t.iloc[:300], t.iloc[300:], ["clock"])
    assert each.loc["clock", "shift"] == "fail"


# ------------------------------------------------ the model and the job
def test_the_ceiling_check_reviews_then_fails():
    y = np.array([0, 0, 1, 1])
    for p, verdict in (([0.1, 0.9, 0.2, 0.8], "pass"),
                       ([0.1, 0.2, 0.9, 0.8], "fail")):
        scored = pd.DataFrame({"not_renewed": y, "model": p})
        assert ceiling_check(scored)[1] == verdict
    scored = pd.DataFrame({"not_renewed": y, "model": [.1, .2, .9, .8]})
    assert ceiling_check(scored, ceiling=1.0 - MARGIN / 2)[1] == "review"


def test_run_checks_stops_on_a_leak_and_allow_lets_it_through(
        warehouse):
    t = rows()
    recent = rows(200, seed=5)
    with pytest.raises(LeakageError, match="the_label: screen"):
        run_checks(t, recent, ["the_label"], sources(t), marks=MARKS,
                   warehouse=warehouse)
    report = run_checks(t, recent, ["the_label"], sources(t),
                        marks=MARKS, warehouse=warehouse,
                        allow={"the_label": "a test, on purpose"})
    assert report.failures() == []
    assert "allowed" in report.text()


def test_a_report_lists_reviews_without_failing():
    sheet = pd.DataFrame({"source": ["orders"], "kind": ["event log"],
                          "moved": [0.0], "audit": ["pass"],
                          "auc": [0.79], "screen": ["review"],
                          "shift auc": [0.5], "shift": ["pass"]},
                         index=pd.Index(["x"], name="column"))
    r = Report(sheet, 0.5, model_auc=CEILING_AUC, model_verdict="review")
    assert r.failures() == []
    assert len(r.reviews()) == 2
