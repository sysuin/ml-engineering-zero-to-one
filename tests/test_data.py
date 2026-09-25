"""
Chapter 23's data tests on the training table as it stands on disk:
its schema against the manifest written beside it, its freshness (was
it built from the warehouse that is here now, by the builder that is
here now?), its row count counted a second way, Chapter 4's structural
checks, and each of Chapter 5's expectations as a test of its own, so
that a failure names the rule. The feeds behind the table are tested
against their contracts in test_contracts.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from foresight import config
from foresight.data import build_table
from foresight.data.expectations import EXPECTATIONS

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def manifest(real_table):
    path = build_table.TABLE.with_suffix(".manifest.json")
    return json.loads(path.read_text())


def test_every_column_has_the_type_it_was_built_with(real_table,
                                                     manifest):
    types = {c: str(t) for c, t in real_table.dtypes.items()}
    assert types == manifest["columns"]
    assert list(real_table.columns) == build_table.COLUMNS


def test_the_content_is_what_the_manifest_says(real_table, manifest):
    rows = pd.util.hash_pandas_object(real_table, index=False)
    digest = hashlib.sha256(rows.values.tobytes()).hexdigest()
    assert digest == manifest["content_sha256"]


def test_it_was_built_from_the_warehouse_here_now(manifest):
    assert build_table.sha256(config.ML_WAREHOUSE) \
        == manifest["warehouse_sha256"]


def test_it_was_built_by_the_builder_here_now(manifest):
    assert build_table.sha256(Path(build_table.__file__)) \
        == manifest["builder_sha256"]


def test_it_has_every_row_the_warehouse_says_it_should(real_table,
                                                       manifest):
    first, last = manifest["contracts_ending"]
    assert len(real_table) == build_table.expected_rows(first, last)


def test_it_passes_chapter_4s_checks(real_table, manifest):
    first, last = manifest["contracts_ending"]
    assert build_table.check(real_table, first, last) == []


@pytest.mark.parametrize("e", EXPECTATIONS, ids=lambda e: e.rule)
def test_every_expectation_holds(real_table, e):
    holds = e.holds(real_table.set_index("contract_id", drop=False))
    broken = holds.index[~holds.astype(bool).to_numpy()]
    assert list(broken[:3]) == []
