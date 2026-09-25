"""
Shared fixtures for the test suite, and the rule that sorts it.

Every test belongs to exactly one of four layers (Chapter 23):

    unit      one function, on inputs small enough to work out by hand
    data      the tables themselves: the warehouse's feeds against their
              contracts, the training table against its checks and
              Chapter 5's expectations, one definition against another
    model     what a fitted model learned: that it ranks, repeats, keeps
              to its floors overall and per slice, and behaves as the
              business expects when one input moves
    service   what runs around the model: the registry, the monthly job,
              the scores database and the API, against their contracts

A test file names its layer with `pytestmark`, and a test with a layer
marker of its own belongs to that layer instead. The files written before Chapter 23 are sorted here instead,
by LAYER and EXCEPT below, because earlier chapters print some of them
line by line. A test with no layer, or with two, stops the run.

A test is also `slow` when it needs the generated Meridian dataset, which
is any test that asks for one of DATA_FIXTURES or is named in
NEEDS_DATA. `make test` runs the rest in seconds, on every push;
`make test-all` runs everything, nightly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from foresight import config  # noqa: E402

LAYERS = {
    "unit": "one function, on inputs small enough to work out by hand",
    "data": "the tables: feeds, the training table, its expectations",
    "model": "what a fitted model learned and how it behaves",
    "service": "the registry, the job, the database and the API",
}
SLOW = "needs the generated dataset, or fits on the real table"

# Files written before Chapter 23: the layer of most of their tests.
LAYER = {
    "test_anomaly": "unit", "test_boosting": "unit",
    "test_build_table": "unit", "test_costs": "unit",
    "test_decide": "unit", "test_evaluate": "unit",
    "test_expectations": "unit", "test_explain": "unit",
    "test_features": "unit", "test_forecast": "unit",
    "test_forest": "unit", "test_impact": "unit",
    "test_leakage": "unit", "test_linear": "unit",
    "test_logistic": "unit", "test_mlp": "unit",
    "test_monitor": "unit", "test_pipeline": "unit",
    "test_regularised": "unit", "test_segments": "unit",
    "test_serve": "service", "test_spend": "unit",
    "test_tracking": "unit", "test_triage": "unit",
}

# ... and the tests in those files that belong to another layer.
EXCEPT = {
    "model": [
        "test_anomaly::test_an_outlier_is_isolated_in_fewer_splits",
        "test_boosting::test_a_longer_gap_never_lowers_the_risk",
        "test_boosting::test_ranks_leavers_above_renewers_on_unseen_rows",
        "test_boosting::test_the_same_booster_twice",
        "test_forest::test_a_different_seed_grows_a_different_forest",
        "test_forest::test_ranks_leavers_above_renewers_on_unseen_rows",
        "test_forest::test_scaling_a_column_changes_nothing",
        "test_forest::test_the_same_forest_twice",
        "test_logistic::test_the_model_learns_the_planted_pattern",
        "test_mlp::test_an_epoch_lowers_the_loss_on_a_learnable_problem",
        "test_mlp::test_fit_predicts_probabilities_and_repeats_itself",
        "test_monitor::test_the_gate_promotes_a_list_that_knows_the_answer",
        "test_monitor::test_the_gate_waits_and_refuses_a_copy",
        "test_pipeline::test_pipeline_scores_as_chapter_16",
        "test_regularised::test_ridge_shrinks_and_lasso_zeroes",
        "test_segments::test_the_segmenter_is_deterministic_and_profiles_add_up",
        "test_triage::test_tfidf_model_learns_and_repeats_itself",
        "test_triage::test_transformer_trains_and_repeats_itself",
    ],
    "data": [
        "test_build_table::test_a_slice_of_the_real_table_passes_its_checks",
        "test_expectations::test_the_real_table_meets_every_expectation",
        "test_forecast::test_sanitation_is_missing_before_its_launch",
        "test_forecast::test_the_panel_starts_when_every_account_is_on_record",
        "test_pipeline::test_shared_definition_is_chapter_4s_table",
        "test_serve::test_the_api_makes_the_rows_the_job_makes",
    ],
    "service": [
        "test_pipeline::test_artifact_never_overwrites",
        "test_pipeline::test_artifact_refuses_a_changed_file",
        "test_pipeline::test_artifact_refuses_another_scikit_learn",
        "test_pipeline::test_artifact_round_trip",
        "test_pipeline::test_inputs_are_checked",
        "test_pipeline::test_registry_lifecycle",
        "test_pipeline::test_registry_refuses_a_version_it_lacks",
    ],
}
LAYER_OF = {name: layer for layer, names in EXCEPT.items()
            for name in names}

# Tests that need the dataset but check for it themselves.
NEEDS_DATA = {
    "test_build_table::test_a_slice_of_the_real_table_passes_its_checks",
    "test_expectations::test_the_real_table_meets_every_expectation",
    "test_forecast::test_sanitation_is_missing_before_its_launch",
    "test_forecast::test_the_panel_starts_when_every_account_is_on_record",
    "test_pipeline::test_shared_definition_is_chapter_4s_table",
}
DATA_FIXTURES = {"warehouse", "ml_warehouse", "real_table"}


def pytest_configure(config: pytest.Config) -> None:
    for name, about in {**LAYERS, "slow": SLOW}.items():
        config.addinivalue_line("markers", f"{name}: {about}")


def key(item: pytest.Item) -> str:
    """test_file::test_name, without any parameters."""
    return f"{item.module.__name__}::{item.originalname}"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    """Give every test its layer, mark the slow ones, and refuse a test
    that has no layer or more than one."""
    wrong = []
    for item in items:
        own = [m.name for m in item.own_markers if m.name in LAYERS]
        have = own or [m for m in LAYERS if item.get_closest_marker(m)]
        if not have:
            layer = LAYER_OF.get(key(item),
                                 LAYER.get(item.module.__name__))
            if layer:
                item.add_marker(getattr(pytest.mark, layer))
                have = [layer]
        if len(have) != 1:
            wrong.append(f"{item.nodeid}: layers {have or 'none'}")
        if (key(item) in NEEDS_DATA
                or DATA_FIXTURES & set(item.fixturenames)):
            item.add_marker(pytest.mark.slow)
    if wrong:
        raise pytest.UsageError("every test needs exactly one layer:\n"
                                + "\n".join(wrong))


@pytest.fixture(scope="session")
def root() -> Path:
    """The repository root, so tests read paths the way the book writes them."""
    return ROOT


@pytest.fixture
def seed() -> int:
    """The book's one seed. Every generator is seeded before each test that asks for it."""
    config.seed_everything(config.SEED)
    return config.SEED


@pytest.fixture(scope="session")
def warehouse() -> Path:
    """Meridian's SQLite warehouse, or a skip that says how to make it."""
    if not config.WAREHOUSE.exists():
        pytest.skip("the Meridian dataset is not generated: run `make data`")
    return config.WAREHOUSE


@pytest.fixture(scope="session")
def real_table():
    """Chapter 4's training table as built from the real warehouse, or a
    skip that says how to make it."""
    import pandas as pd

    from foresight.data.build_table import TABLE
    if not config.ML_WAREHOUSE.exists() or not TABLE.exists():
        pytest.skip("build the table first: make data, then"
                    " python -m foresight.data.build_table")
    return pd.read_parquet(TABLE)


@pytest.fixture(scope="session")
def ml_warehouse() -> Path:
    """Meridian's machine-learning warehouse, or a skip."""
    if not config.ML_WAREHOUSE.exists():
        pytest.skip("the Meridian dataset is not generated: run `make data`")
    return config.ML_WAREHOUSE
