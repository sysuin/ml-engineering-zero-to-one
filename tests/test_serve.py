"""
Chapter 22's serving code: reasons that add up for the pipeline's
model; the list's ranking, key accounts apart; the rule; a scores
database that keeps every run and lets a cohort finish once; a lock
that holds, lets go after a failure and is taken over from a dead
run; and an API whose contract refuses bad requests before anything
runs, falls back to the rule, and answers 503 rather than inventing a
number. Made-up tables and temporary folders; the tests that need the
warehouse skip without it.
"""
from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight import config
from foresight.explain import facts, log_odds
from foresight.pipeline.artifact import schema
from foresight.pipeline.model import INPUTS, build
from foresight.serve import batch, renewal, store

pytestmark = pytest.mark.filterwarnings("ignore:.*httpx.*")


@pytest.fixture(scope="module")
def rows():
    return made_up(cohorts=10, per_cohort=80)


@pytest.fixture(scope="module")
def fitted(rows):
    return build().fit(rows, rows.not_renewed)


def named(rows, key=()):
    return rows.assign(name=[f"Account {i}" for i in rows.account_id],
                       account_manager="A. Person",
                       is_key_account=rows.contract_id.isin(key)
                       .astype(int))


# ------------------------------------------------ renewal.py
def test_contributions_add_up_to_the_log_odds(rows, fitted):
    base, parts = renewal.contributions(fitted, rows)
    p = fitted.predict_proba(rows)[:, 1]
    assert np.allclose(base + parts.sum(axis=1), log_odds(p))
    assert set(parts.columns) == set(INPUTS)


def test_every_model_column_has_a_source(fitted):
    names = fitted.estimator_.named_steps["prepare"] \
        .get_feature_names_out()
    assert {renewal.source(n) for n in names} == set(INPUTS)


def test_explain_gives_at_most_three_reasons(rows, fitted):
    out = renewal.explain(fitted, rows.head(50), rows)
    assert out.reasons.map(len).max() <= 3
    assert out.chance.between(0, 1).all()


def test_key_accounts_are_listed_apart_without_a_chance(rows, fitted):
    cohort = named(rows[rows.moment == rows.moment.max()],
                   key=rows.contract_id.head(700))
    out = renewal.rank(renewal.explain(fitted, cohort, rows), k=5)
    keys = out[out.is_key_account == 1]
    assert keys.chance.isna().all() and not keys.listed.any()
    tail = out[out.is_key_account == 0]
    assert list(tail["rank"]) == list(range(1, len(tail) + 1))
    assert tail.chance.is_monotonic_decreasing
    assert tail.listed.sum() == min(5, len(tail))


def test_rule_puts_the_longest_gap_first_and_no_order_last(rows):
    cohort = named(rows.head(6).copy())
    cohort["days_since_order"] = pd.array([30, None, 200, 5, 90, 1],
                                          dtype="Int64")
    out = renewal.rule(cohort, k=3)
    assert list(out.days_since_order.head(3)) == [200, 90, 30]
    assert pd.isna(out.days_since_order.iloc[-1])
    assert out.chance.isna().all()


# ------------------------------------------------ store.py
def test_a_cohort_finishes_once_unless_a_reason_is_given(tmp_path):
    with closing(store.connect(tmp_path / store.FILE)) as con:
        first = store.start(con, "renewal", "2025-12-31", "2025-12-31")
        with con:
            con.execute("UPDATE runs SET status = 'finished'"
                        " WHERE run = ?", (first,))
        assert store.finished(con, "renewal", "2025-12-31") == first
        second = store.start(con, "renewal", "2025-12-31",
                             "2026-01-02")
        with pytest.raises(sqlite3.IntegrityError), con:
            con.execute("UPDATE runs SET status = 'finished'"
                        " WHERE run = ?", (second,))
        third = store.start(con, "renewal", "2025-12-31", "2026-01-02",
                            reason="a corrected column")
        with con:
            con.execute("UPDATE runs SET status = 'finished'"
                        " WHERE run = ?", (third,))
        assert store.finished(con, "renewal", "2025-12-31") == first


def test_a_failed_run_keeps_its_step_and_its_error(tmp_path):
    with closing(store.connect(tmp_path / store.FILE)) as con:
        run = store.start(con, "renewal", "2025-12-31", "2025-12-31")
        store.step(con, run, "holdout")
        store.fail(con, run, OSError("disk full"))
        r = con.execute("SELECT status, step, error"
                        " FROM runs").fetchone()
    assert r == ("failed", "holdout", "OSError: disk full")


def test_scores_and_inputs_are_written_together(tmp_path, rows,
                                                fitted):
    cohort = rows[rows.moment == rows.moment.max()]
    scored = renewal.rank(renewal.explain(fitted, named(cohort), rows))
    scored["arm"] = None
    file = tmp_path / store.FILE
    with closing(store.connect(file)) as con:
        run = store.start(con, "renewal", cohort.moment.iloc[0],
                          cohort.moment.iloc[0])
        store.write(con, run, cohort.moment.iloc[0], scored, cohort)
    kept = store.inputs(file, run, {"inputs": schema(rows, INPUTS)})
    again = fitted.predict_proba(kept)[:, 1]
    answers = store.read(file, "SELECT contract_id, chance FROM"
                         " answers ORDER BY contract_id")
    assert np.allclose(again, answers.chance)
    status = store.read(file, "SELECT status FROM runs").status[0]
    assert status == "finished"


# ------------------------------------------------ batch.py: the lock
def test_a_second_run_is_refused_while_the_lock_is_held(tmp_path):
    with batch.lock(tmp_path / batch.LOCK):
        with pytest.raises(batch.Locked):
            with batch.lock(tmp_path / batch.LOCK):
                pass
    assert not (tmp_path / batch.LOCK).exists()


def test_the_lock_is_released_after_a_failure(tmp_path):
    with pytest.raises(ValueError):
        with batch.lock(tmp_path / batch.LOCK):
            raise ValueError("step 3")
    assert not (tmp_path / batch.LOCK).exists()


def test_a_dead_runs_lock_is_taken_over(tmp_path):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    file = tmp_path / batch.LOCK
    file.write_text(f"{socket.gethostname()} {gone.pid}")
    assert batch.stale(file)
    with batch.lock(file):
        assert file.read_text().endswith(str(os.getpid()))


def test_a_lock_from_another_machine_is_never_taken(tmp_path):
    file = tmp_path / batch.LOCK
    file.write_text("elsewhere 1")
    assert not batch.stale(file)


# ------------------------------------------------ api.py
@pytest.fixture
def service(tmp_path):
    from foresight.serve.api import Service
    return Service(tmp_path, warehouse=tmp_path / "none.db",
                   on="2025-12-31")


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    from foresight.serve.api import create_app
    return TestClient(create_app(service))


def contract(key=0):
    return pd.Series({"contract_id": 7, "name": "Account 7",
                      "moment": pd.Timestamp("2025-12-31"),
                      "is_key_account": key})


@pytest.mark.parametrize("body", [
    {"contract_id": "7"}, {"contract_id": 7.5}, {"contract_id": 0},
    {"contract": 7}, {"contract_id": 7, "extra": 1}, {}])
def test_the_request_is_a_contract(client, service, body):
    service.score = lambda *a: pytest.fail("the service was asked")
    assert client.post("/renewal", json=body).status_code == 422


@pytest.mark.parametrize("body", ["", "   ", "x" * 2001])
def test_a_ticket_must_have_some_text_and_not_too_much(client, body):
    assert client.post("/triage", json={"body": body}).status_code \
        == 422


def test_without_a_model_the_service_says_degraded(client):
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["renewal"] is None


def test_without_a_model_the_rule_answers(client, service, rows):
    service.contract = lambda cid: contract()
    service.fetch = lambda ids, wh: rows.head(1).assign(
        days_since_order=pd.array([88], dtype="Int64"))
    a = client.post("/renewal", json={"contract_id": 7}).json()
    assert a["answered_by"] == "rule" and a["chance"] is None
    assert a["reasons"] == ["Last order 88 days ago"]


def test_a_key_account_gets_no_chance(client, service):
    service.contract = lambda cid: contract(key=1)
    a = client.post("/renewal", json={"contract_id": 7}).json()
    assert a["answered_by"] == "key account" and a["chance"] is None


def test_an_unknown_segment_falls_back_to_the_rule(client, service,
                                                   rows, fitted):
    service.renewal = {"model": fitted, "name": "renewal v1",
                       "manifest": {"inputs": schema(rows, INPUTS)},
                       "train": rows, "typical": facts(rows)}
    service.contract = lambda cid: contract()
    service.fetch = lambda ids, wh: rows.head(1)
    a = client.post("/renewal", json={"contract_id": 7}).json()
    assert a["answered_by"] == "model" and 0 < a["chance"] < 1
    service.fetch = lambda ids, wh: rows.head(1).assign(
        segment="Wholesale")
    a = client.post("/renewal", json={"contract_id": 7}).json()
    assert a["answered_by"] == "rule"
    assert "Wholesale" in a["note"]


def test_no_warehouse_is_a_503_not_a_number(client):
    r = client.post("/renewal", json={"contract_id": 7})
    assert r.status_code == 503


def test_triage_without_a_model_uses_the_urgent_words(client):
    a = client.post("/triage", json={"body": "need it today"}).json()
    assert a["answered_by"] == "keywords"
    assert a["priority"] == "Urgent" and a["category"] is None


def test_a_forecast_path_must_look_like_one(client):
    assert client.get("/forecast/mops/Midwest").status_code == 422
    assert client.get("/forecast/MRD-CLE-001/Mars").status_code == 422
    assert client.get("/forecast/MRD-CLE-001/Midwest").status_code \
        == 404


# ------------------------------------------------ with the warehouse
@pytest.fixture(scope="module")
def warehouse():
    if not config.ML_WAREHOUSE.exists():
        pytest.skip("the Meridian dataset is not generated: run"
                    " `make data`")
    return config.ML_WAREHOUSE


def test_the_api_makes_the_rows_the_job_makes(warehouse):
    from foresight.pipeline import features
    from foresight.serve import online
    cohort = features.at_mark("2024-10-02")
    features.check_skew(cohort, online.rows(cohort.contract_id))


def test_the_job_refuses_a_mark_the_warehouse_has_not_reached(
        warehouse):
    with pytest.raises(batch.StaleData):
        batch.fresh(pd.Timestamp("2026-01-30"))
    batch.fresh(pd.Timestamp("2025-12-31"))
