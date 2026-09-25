"""
Shared fixtures for the test suite.

The suite grows with Foresight: unit tests for feature code, data tests on the training
table, model tests per slice, and the CI gate that compares a retrained model with the one
in production (Chapter 23's four layers). Today it holds only what every layer will need —
the project on the import path, the book's seed, and a way to skip a test cleanly when the
generated dataset is not there, rather than failing with a missing file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from foresight import config  # noqa: E402


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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """
    An empty suite passes. pytest exits 5 when it collects nothing, which would fail CI
    for every chapter before Foresight has code to test. Remove this once the first test
    exists: from then on, collecting nothing is a real failure.
    """
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
