"""
Chapter 27's demonstration: the beats in the order the chapter promises
(it works, it fails correctly, it shows the evidence); a workspace that
copies the sandbox's registry and nothing else, never writes to the
sandbox, and trains its own models only when the sandbox has none; and
output that fits the page. Temporary folders; no model is fitted.
"""
from __future__ import annotations

import hashlib
import shutil

import pandas as pd
import pytest

from foresight import demo

pytestmark = pytest.mark.service


def tree(folder) -> dict:
    return {str(p.relative_to(folder)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture
def sandbox(tmp_path):
    """A sandbox with both registries, a scores database and a
    holdout: everything a copy must leave behind."""
    box = tmp_path / "sandbox"
    for model in demo.MODELS:
        (box / "artifacts" / model / "1").mkdir(parents=True)
        (box / "artifacts" / model / "registry.json").write_text("{}")
        (box / "artifacts" / model / "1" / "model.joblib").write_bytes(
            model.encode())
    (box / "data" / "foresight").mkdir(parents=True)
    (box / "data" / "foresight" / "scores.db").write_bytes(b"runs")
    (box / "data" / "foresight" / "holdout.csv").write_text("arm\n")
    return box


def test_the_beats_run_in_the_order_the_chapter_gives():
    names = [b.__name__ for b in demo.BEATS]
    assert names == ["the_list", "the_forecast", "the_api",
                     "the_monitor", "the_mcp", "the_failures",
                     "the_evidence"]


def test_the_workspace_copies_the_registry_and_nothing_else(sandbox):
    before = tree(sandbox)
    root = demo.workspace(sandbox)
    try:
        assert root != sandbox and not root.is_relative_to(sandbox)
        assert tree(root) == {k: v for k, v in before.items()
                              if k.startswith("artifacts/")}
        assert not (root / "data").exists()
        assert tree(sandbox) == before
    finally:
        shutil.rmtree(root)


def test_a_sandbox_without_models_means_training_in_the_workspace(
        sandbox, monkeypatch):
    shutil.rmtree(sandbox / "artifacts" / "triage")
    trained = []
    monkeypatch.setattr(demo, "train", trained.append)
    root = demo.workspace(sandbox)
    try:
        assert trained == [root]
        assert not (root / "artifacts").exists()
    finally:
        shutil.rmtree(root)


def test_what_it_says_fits_the_page(tmp_path, capsys):
    d = demo.Demo(tmp_path, pd.Timestamp(demo.MORNING))
    d.heading(4, "The monitor", "the weekly page")
    d.heading(6, "It fails correctly", "")
    d.say("word " * 60, "short")
    printed = capsys.readouterr().out.splitlines()
    assert all(len(line) <= demo.WIDTH for line in printed)
    assert printed[1].endswith("the weekly page")
    assert printed[1] == printed[1].rstrip()
    assert printed[3] == "6  It fails correctly"
    assert d.lines[-1] == "   short"
