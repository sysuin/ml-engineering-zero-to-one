"""
Chapter 23's model tests. The model the settings file describes,
backtested on the cohorts its floors were set on, must keep every
floor in tests/model_floors.json: ahead of Chapter 3's rule, precision
at capacity and AUC above the lower ends of v0.6's intervals, and each
slice's share of its leavers called above the same. The floors come
from listing 23/05, never from a hope; a floor of zero tests nothing
and is skipped, saying so.
"""
from __future__ import annotations

import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight import floors, gate
from foresight.config import rng
from foresight.evaluate import backtest

pytestmark = pytest.mark.model
FLOORS = floors.read()
SLICES = [(by, name) for by, kept in FLOORS["slices"].items()
          for name in kept]


@pytest.fixture(scope="module")
def scored(real_table):
    model, _ = gate.candidate()
    return gate.scores(model, real_table, FLOORS["as_of"],
                       FLOORS["cohorts"])


def test_the_list_is_not_behind_the_rule(scored):
    assert not [s for s in floors.shortfalls(scored, FLOORS)
                if "rule" in s]


@pytest.mark.parametrize("what", ["precision", "auc"])
def test_overall_floors(scored, what):
    assert not [s for s in floors.shortfalls(scored, FLOORS)
                if s.startswith(what)]


@pytest.mark.parametrize("by, name", SLICES,
                         ids=[f"{b}-{n}" for b, n in SLICES])
def test_every_slice_keeps_its_floor(scored, by, name):
    least = FLOORS["slices"][by][name]
    if least == 0:
        pytest.skip(f"{by} {name}: a floor of zero tests nothing")
    assert floors.called_share(scored, by)[name] >= least


@pytest.mark.unit
def test_a_reference_always_keeps_the_floors_it_set():
    t = made_up(cohorts=6, per_cohort=150)
    s = backtest(t, "2023-01-01", "2023-06-30",
                 lambda: Reference()).assign(
        size="smallest", tenure="1-3 years", population="long tail")
    f = floors.set_floors(s, reps=200)
    assert floors.shortfalls(s, f) == []
    worse = s.assign(model=rng(5).random(len(s)))
    assert floors.shortfalls(worse, f)


class Reference:
    """Scores by the gap, as the made-up label is drawn."""

    def fit(self, rows):
        return self

    def predict_proba(self, rows):
        return rows.days_since_order.astype(float).fillna(365) \
            .to_numpy()


def test_the_floors_are_v06s():
    assert FLOORS["reference"] == "v0.6"
    assert isinstance(FLOORS["precision"], float)
    assert pd.Timestamp(FLOORS["as_of"]) == pd.Timestamp("2025-01-01")
