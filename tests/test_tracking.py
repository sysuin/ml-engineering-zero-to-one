"""
Chapter 15's tracking and tuning: a run's record names its table, its
rows, its code and its scores; the same run twice has the same id; the
log is only ever appended to; MLflow gets the same record and the
model; a tuning search keeps to its budget, repeats itself, and reads
its cache back exactly. Made-up tables and a temporary store only.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight import tracking, tune
from foresight.models.boosting import RenewalBooster

# MLflow's database layer warns about its own use of SQLAlchemy.
quiet_mlflow = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def stored(tmp_path):
    """A small table written with a manifest, as Chapter 4 writes one."""
    t = made_up(cohorts=20, per_cohort=60)     # ends Jan 2023-Aug 2024
    path = tmp_path / "renewals_table.parquet"
    t.to_parquet(path, index=False)
    path.with_suffix(".manifest.json").write_text(json.dumps(
        {"content_sha256": tracking.rows_hash(t)}))
    return t, path


def rows_of(t):
    return t[t.end_date.between(*tune.TRAINING)]


# ------------------------------------------------ the record
def test_the_table_hash_matches_its_manifest(stored):
    t, path = stored
    assert tracking.table_version(None, path)["matches_manifest"]


def test_one_changed_value_breaks_the_match(stored):
    t, path = stored
    copy = t.copy()
    copy.loc[0, "spend_365"] += 0.01
    v = tracking.table_version(copy, path)
    assert not v["matches_manifest"]
    assert v["content_sha256"] != tracking.rows_hash(t)


def test_the_rows_a_model_saw_are_recorded(stored):
    t, path = stored
    rows = rows_of(t)
    d = tracking.data_record(rows, t, path)
    assert d["rows"] == len(rows)
    assert d["rows_sha256"] == tracking.rows_hash(rows)
    assert d["rows_sha256"] != d["content_sha256"]


def test_the_same_run_has_the_same_id_whenever_it_runs():
    a = tracking.record("x", {"k": 1}, {"d": "h"}, {"c": "h"},
                        {"log loss": 0.2})
    b = tracking.record("x", {"k": 1}, {"d": "h"}, {"c": "h"},
                        {"log loss": 0.2})
    c = tracking.record("x", {"k": 1}, {"d": "h"}, {"c": "h"},
                        {"log loss": 0.21})
    assert a["run"] == b["run"] != c["run"]
    assert "when" in a


def test_code_hash_follows_the_contents(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("RATE = 0.03\n")
    before = tracking.code_hash([f])
    f.write_text("RATE = 0.3\n")
    assert tracking.code_hash([f]) != before


def test_the_files_behind_a_tuned_booster():
    names = sorted(p.name for p in
                   tracking.source_files(tune.TunableBooster()))
    assert names == ["boosting.py", "config.py", "tune.py"]


def test_the_log_is_only_appended_to(tmp_path):
    log = tracking.RunLog(tmp_path / "runs.jsonl")
    first = log.append({"run": "a", "n": 1})
    log.append({"run": "b", "n": 2})
    lines = log.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == first
    assert [r["run"] for r in log.runs()] == ["a", "b"]


# ------------------------------------------------ MLflow
@quiet_mlflow
def test_mlflow_gets_the_record_and_the_model(stored, tmp_path):
    t, path = stored
    rows = rows_of(t)
    model = RenewalBooster(min_leaf=10, most=300, patience=30).fit(rows)
    rec = tracking.record("small", {"leaves": 2},
                          tracking.data_record(rows, t, path),
                          tracking.code_record(model),
                          {"log loss": 0.2, "auc": 0.7})
    store = tmp_path / "mlruns"
    tracking.log_mlflow(rec, model, store)
    found = tracking.runs(store)
    assert len(found) == 1
    r = found.iloc[0]
    assert r["tags.mlflow.runName"] == "small"
    assert r["params.leaves"] == "2"
    assert r["metrics.log_loss"] == pytest.approx(0.2)
    assert r["tags.run"] == rec["run"]
    assert r["tags.data.content_sha256"] == rec["data"]["content_sha256"]
    mlflow = tracking.open_store(store)
    client = mlflow.MlflowClient()
    names = {a.path for a in client.list_artifacts(r.run_id)}
    assert names == {"model.pkl", "record.json"}
    local = client.download_artifacts(r.run_id, "model.pkl",
                                      str(tmp_path))
    with open(local, "rb") as f:
        again = pickle.load(f)
    assert np.array_equal(again.predict_proba(rows),
                          model.predict_proba(rows))


@quiet_mlflow
def test_track_writes_both_records(stored, tmp_path):
    t, path = stored
    rows = rows_of(t)
    model = RenewalBooster(min_leaf=10, most=300, patience=30).fit(rows)
    rec = tracking.track("both", model, {"leaves": 2}, rows, t,
                         {"log loss": 0.2}, store=tmp_path / "mlruns",
                         log=tmp_path / "runs.jsonl", table_path=path)
    logged = tracking.RunLog(tmp_path / "runs.jsonl").runs()
    assert [r["run"] for r in logged] == [rec["run"]]
    assert rec["data"]["matches_manifest"]
    assert len(tracking.runs(tmp_path / "mlruns")) == 1


# ------------------------------------------------ tuning
def test_chapter_11_settings_are_chapter_11s_booster():
    t = made_up(cohorts=12, per_cohort=100)
    a = RenewalBooster(min_leaf=10).fit(t).predict_proba(t)
    b = tune.TunableBooster(**{**tune.CHAPTER_11, "min_leaf": 10}
                            ).fit(t).predict_proba(t)
    assert np.array_equal(a, b)


def test_random_draws_repeat_and_stay_in_range():
    a, b = tune.draw(20), tune.draw(20)
    assert a == b
    for s in a:
        for name, (lo, hi, _, whole) in tune.SPACE.items():
            assert lo <= s[name] <= hi
            assert isinstance(s[name], int) == whole


def test_a_grid_is_every_combination():
    g = tune.grid({"min_leaf": [5, 45, 400], "rate": [0.02, 0.3]})
    assert len(g) == 6
    assert {"min_leaf": 400, "rate": 0.3} in g


def fake_score(table, settings=None, **kw):
    """A quick objective: best near two leaves and a rate of 0.1."""
    s = settings or {}
    return {"log loss": 0.2 + 0.001 * abs(np.log(s["leaves"] / 2))
            + 0.01 * abs(np.log(s["rate"] / 0.1)),
            "auc": 0.7, "leavers": 10}


def test_a_search_keeps_to_its_budget_and_repeats(monkeypatch,
                                                  tmp_path):
    monkeypatch.setattr(tune, "score", fake_score)
    monkeypatch.setattr(tune, "CACHE", tmp_path)
    t = made_up(cohorts=4, per_cohort=10)
    capped = tune.search(t, tune.Budget(12, 100), cache=False)
    assert len(capped) == 12
    assert capped.attrs["stopped by"] == "trials"
    again = tune.search(t, tune.Budget(12, 100), cache=False)
    pd.testing.assert_frame_equal(capped, again)
    patient = tune.search(t, tune.Budget(200, 5), cache=False)
    assert patient.attrs["stopped by"] == "patience"
    assert len(patient) < 200


def test_the_cache_gives_back_exactly_what_was_searched(monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(tune, "score", fake_score)
    monkeypatch.setattr(tune, "CACHE", tmp_path)
    t = made_up(cohorts=4, per_cohort=10)
    first = tune.search(t, tune.Budget(8, 100))
    assert len(list((tmp_path / "tune").iterdir())) == 1
    second = tune.search(t, tune.Budget(8, 100))
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert tune.best(first) == tune.best(second)
    other = tune.search(t, tune.Budget(8, 100), seed=tune.SEED + 1)
    assert len(list((tmp_path / "tune").iterdir())) == 2
    assert not other.equals(first)
