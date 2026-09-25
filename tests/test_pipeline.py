"""
Chapter 21's pipeline: the scikit-learn model scores as Chapter 16's
does; the skew test finds a changed value and nothing else; an
artifact refuses a changed file or another scikit-learn; the registry
promotes, archives and rolls back without losing a version; settings
refuse a key they do not have. Made-up tables and temporary folders;
one test reads the warehouse, and skips without it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight import config
from foresight.decide import Calibrated
from foresight.models.featured import FeaturedLasso
from foresight.pipeline import artifact, features, settings
from foresight.pipeline.model import INPUTS, build, weights
from foresight.pipeline.registry import Registry, RegistryError


@pytest.fixture(scope="module")
def rows():
    return made_up(cohorts=10, per_cohort=80)


@pytest.fixture(scope="module")
def fitted(rows):
    return build().fit(rows, rows.not_renewed)


# ------------------------------------------------ the model
def test_pipeline_scores_as_chapter_16(rows, fitted):
    old = Calibrated(lambda: FeaturedLasso((), 0.002)).fit(rows)
    new = fitted.predict_proba(rows)[:, 1]
    assert np.abs(old.predict_proba(rows) - new).max() < 1e-9


def test_sixteen_named_numbers(fitted):
    w = weights(fitted)
    assert len(w) == 16
    assert "days_since_order" in w and "region_West" in w
    assert "segment_Small business" not in w        # the baseline


def test_unknown_category_is_an_error(rows, fitted):
    odd = rows.head(3).assign(segment="Wholesale")
    with pytest.raises(ValueError):
        fitted.predict_proba(odd)


def test_missing_segment_counts_as_small_business(rows, fitted):
    a = rows.head(5).assign(segment="Small business")
    b = rows.head(5).assign(segment=np.nan)
    assert np.allclose(fitted.predict_proba(a), fitted.predict_proba(b))


def test_fit_does_not_read_the_label_column(rows):
    """Everything but the label goes in; a model that could see
    not_renewed among its inputs would score perfectly."""
    X = rows.drop(columns="not_renewed")
    a = build().fit(X, rows.not_renewed).predict_proba(X)
    b = build().fit(rows, rows.not_renewed).predict_proba(rows)
    assert np.allclose(a, b)
    assert "not_renewed" not in INPUTS


# ------------------------------------------------ the skew test
def test_skew_is_zero_for_the_same_rows(rows):
    found = features.skew(rows, rows.sample(frac=1, random_state=0))
    assert found.differ.sum() == 0


def test_skew_finds_one_changed_value(rows):
    other = rows.copy()
    other.loc[7, "orders_90d"] += 1
    found = features.skew(rows, other)
    assert found.differ.sum() == 1
    assert found.loc["orders_90d", "example"] == rows.contract_id[7]
    with pytest.raises(features.SkewError, match="orders_90d"):
        features.check_skew(rows, other)


def test_skew_gap_matches_only_a_gap(rows):
    other = rows.copy()
    other["days_since_order"] = other.days_since_order.astype("Float64")
    other.loc[3, "days_since_order"] = pd.NA
    assert features.skew(rows, other).loc["days_since_order",
                                          "differ"] == 1


def test_skew_tolerates_rounding_only(rows):
    other = rows.assign(spend_365=rows.spend_365 + 0.004)
    assert features.skew(rows, other).differ.sum() == 0
    other = rows.assign(spend_365=rows.spend_365 + 0.02)
    assert features.skew(rows, other).loc["spend_365", "differ"] == len(rows)


# ------------------------------------------------ the artifact
def manifest(rows):
    return {"name": "renewal", "run": "test", "data": {"as_of": "x"},
            "inputs": artifact.schema(rows, INPUTS),
            "metrics": {"model hits": 1, "model auc": 0.5}}


def test_artifact_round_trip(tmp_path, rows, fitted):
    artifact.save(tmp_path / "a", fitted, manifest(rows), "# card\n")
    model, m = artifact.load(tmp_path / "a")
    assert np.array_equal(model.predict_proba(rows),
                          fitted.predict_proba(rows))
    assert set(m["files"]) == {"model.joblib", "model_card.md"}


def test_artifact_never_overwrites(tmp_path, rows, fitted):
    artifact.save(tmp_path / "a", fitted, manifest(rows))
    with pytest.raises(FileExistsError):
        artifact.save(tmp_path / "a", fitted, manifest(rows))


def test_artifact_refuses_a_changed_file(tmp_path, rows, fitted):
    artifact.save(tmp_path / "a", fitted, manifest(rows), "# card\n")
    (tmp_path / "a" / "model_card.md").write_text("# edited\n")
    with pytest.raises(artifact.ArtifactError, match="model_card"):
        artifact.load(tmp_path / "a")


def test_artifact_refuses_another_scikit_learn(tmp_path, rows, fitted):
    artifact.save(tmp_path / "a", fitted, manifest(rows))
    path = tmp_path / "a" / "manifest.json"
    m = json.loads(path.read_text())
    m["environment"]["scikit-learn"] = "0.0.1"
    path.write_text(json.dumps(m))
    with pytest.raises(artifact.ArtifactError, match="scikit-learn"):
        artifact.load(tmp_path / "a")


def test_inputs_are_checked(rows):
    m = manifest(rows)
    artifact.check_inputs(rows, m)
    with pytest.raises(artifact.ArtifactError, match="missing"):
        artifact.check_inputs(rows.drop(columns="region"), m)
    with pytest.raises(artifact.ArtifactError, match="discount_pct"):
        artifact.check_inputs(
            rows.assign(discount_pct=rows.discount_pct.astype(float)), m)


# ------------------------------------------------ the registry
def test_registry_lifecycle(tmp_path, rows, fitted):
    reg = Registry("renewal", tmp_path)
    v1 = reg.register(fitted, manifest(rows))
    v2 = reg.register(fitted, manifest(rows))
    assert (v1, v2) == (1, 2)
    stages = lambda: {v: r["stage"]                          # noqa: E731
                      for v, r in reg.read()["versions"].items()}
    assert stages() == {"1": "staged", "2": "staged"}
    with pytest.raises(RegistryError):
        reg.production()
    reg.promote(v1, "first")
    reg.promote(v2, "second")
    assert stages() == {"1": "archived", "2": "production"}
    assert reg.rollback("bad list") == 1
    assert stages() == {"1": "production", "2": "archived"}
    with pytest.raises(RegistryError, match="nothing to roll back"):
        reg.rollback("again")
    with pytest.raises(RegistryError, match="not staged"):
        reg.promote(v2, "archived versions stay archived")
    assert len(reg.read()["log"]) == 7
    assert (tmp_path / "renewal" / "2" / "model.joblib").exists()


def test_registry_refuses_a_version_it_lacks(tmp_path):
    with pytest.raises(RegistryError, match="no version"):
        Registry("renewal", tmp_path).promote(3, "missing")


# ------------------------------------------------ settings
def test_settings_load_and_override():
    s = settings.load("renewal", ["data.as_of=2024-12-01",
                                  "evaluation.cohorts=3"])
    assert s["data"]["as_of"] == "2024-12-01"
    assert s["evaluation"]["cohorts"] == 3
    assert settings.load("renewal")["data"]["as_of"] == "2025-01-01"


@pytest.mark.parametrize("bad, error", [
    ("data.as-of=2024-12-01", KeyError), ("model=1", KeyError),
    ("nothing.here=1", KeyError), ("evaluation.cohorts=six", ValueError)])
def test_settings_refuse(bad, error):
    with pytest.raises(error):
        settings.load("renewal", [bad])


def test_settings_bool():
    assert settings.parse("true", False) is True
    with pytest.raises(ValueError):
        settings.parse("yes", False)


# ------------------------------------------------ one definition
def test_shared_definition_is_chapter_4s_table():
    if not config.ML_WAREHOUSE.exists():
        pytest.skip("the Meridian dataset is not generated: run `make data`")
    from foresight.data.build_table import TABLE
    from foresight.tracking import rows_hash
    if not TABLE.exists():
        pytest.skip("Chapter 4's table is not built")
    ours = features.labelled("2023-01-01", "2025-12-31")
    assert rows_hash(ours) == rows_hash(pd.read_parquet(TABLE))
