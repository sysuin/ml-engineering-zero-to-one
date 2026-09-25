"""
Chapter 23's contract tests for what Foresight serves. Two readers
depend on it: the CRM's account page, which reads POST /renewal, and
the account team's dashboard, which reads the answers view. The API's
published schema must still offer every field the CRM reads, with a
type it can handle; every answer actually sent must match; and a list
written by the monthly job's own functions must keep the dashboard's
contract, which a tampered row breaks. Made-up tables and temporary
folders only.
"""
from __future__ import annotations

from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight.contracts import ANSWERS, breaches, read_only
from foresight.contracts.answers import CRM_READS, answered, offered
from foresight.explain import facts
from foresight.pipeline.artifact import schema
from foresight.pipeline.model import INPUTS, build
from foresight.serve import renewal, store

pytestmark = [pytest.mark.service,
              pytest.mark.filterwarnings("ignore:.*httpx.*")]


@pytest.fixture(scope="module")
def rows():
    return made_up(cohorts=10, per_cohort=80)


@pytest.fixture(scope="module")
def fitted(rows):
    return build().fit(rows, rows.not_renewed)


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


# ------------------------------------------------ the CRM
def test_the_schema_still_offers_every_field_the_crm_reads(client):
    spec = client.get("/openapi.json").json()
    assert offered(spec["components"]["schemas"]["RenewalAnswer"]) \
        == []


def test_offered_names_a_field_renamed_or_retyped():
    spec = {"required": ["contract_id"], "properties": {
        "contract_id": {"type": "string"},
        "chance": {"anyOf": [{"type": "number"}, {"type": "null"}]}}}
    problems = offered(spec, {"contract_id": ("integer",),
                              "chance": ("number", "null"),
                              "note": ("string",)})
    assert problems == ["contract_id: now ['string']", "note: gone"]


def test_every_kind_of_answer_matches_the_crm_contract(
        client, service, rows, fitted):
    service.contract = lambda cid: contract()
    service.fetch = lambda ids, wh: rows.head(1)
    bodies = [client.post("/renewal", json={"contract_id": 7}).json()]
    service.renewal = {"model": fitted, "name": "renewal v1",
                       "manifest": {"inputs": schema(rows, INPUTS)},
                       "train": rows, "typical": facts(rows)}
    bodies.append(client.post("/renewal",
                              json={"contract_id": 7}).json())
    service.contract = lambda cid: contract(key=1)
    bodies.append(client.post("/renewal",
                              json={"contract_id": 7}).json())
    assert [b["answered_by"] for b in bodies] == [
        "rule", "model", "key account"]
    for b in bodies:
        assert answered(b) == []


def test_answered_names_a_wrong_type():
    body = {name: None for name in CRM_READS}
    assert "contract_id: NoneType" in answered(body)
    assert "chance: NoneType" not in answered(body)


# ------------------------------------------------ the dashboard
@pytest.fixture
def written(tmp_path, rows, fitted):
    """A list made with the monthly job's own functions: two key
    accounts listed apart, the forty split into called and held out."""
    cohort = rows[rows.moment == rows.moment.max()].copy()
    cohort["name"] = [f"Account {i}" for i in cohort.account_id]
    cohort["account_manager"] = "A. Person"
    cohort["is_key_account"] = cohort.contract_id.isin(
        cohort.contract_id.head(2)).astype(int)
    scored = renewal.rank(renewal.explain(fitted, cohort, rows))
    arms = np.where(scored.contract_id % 2 == 0, "called", "held out")
    scored["arm"] = np.where(scored.listed, arms, None)
    file = tmp_path / store.FILE
    with closing(store.connect(file)) as con:
        mark = cohort.moment.iloc[0]
        run = store.start(con, "renewal", mark, mark)
        store.write(con, run, mark, scored, cohort)
    return file


def test_a_list_keeps_the_dashboards_contract(written):
    with closing(read_only(written)) as con:
        assert breaches(con, ANSWERS) == []


def test_a_key_account_given_a_chance_breaks_it(written):
    with closing(store.connect(written)) as con, con:
        con.execute("UPDATE scores SET chance = 0.5"
                    " WHERE key_account = 1")
    with closing(read_only(written)) as con:
        found = [b.clause for b in breaches(con, ANSWERS)]
    assert found == ["a key account has no chance and no rank"]


def test_the_health_check_says_ok_or_degraded(client):
    health = client.get("/health").json()
    assert health["status"] in ("ok", "degraded")
    assert isinstance(health["problems"], list)
