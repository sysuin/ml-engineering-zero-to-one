"""
Chapter 23's gate: Chapter 24's rule on the overall list, and no
protected slice clearly worse. Made-up cohorts whose answers are known:
a copy is unchanged, a list that knows the answer is promoted, a list
that has forgotten it is blocked, and a list that is level overall but
has given up one region's leavers is blocked on that region. register()
refuses what the gate blocked. One slow test runs the gate on the real
validation cohorts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_evaluate import table as made_up

from foresight import gate
from foresight.config import rng

pytestmark = pytest.mark.unit
REPS = 400


def scored(seed=0) -> pd.DataFrame:
    """Six cohorts of 200, labelled with every protected slice, and a
    champion's chance that knows the answer a little."""
    t = made_up(cohorts=6, per_cohort=200, seed=seed)
    g = rng(seed + 1)
    y = t.not_renewed.to_numpy(float)
    return t.assign(
        model=0.6 * y + g.random(len(t)), rule=0.0,
        size=g.choice(["smallest", "largest"], len(t)),
        tenure="1-3 years", population="long tail")


def test_a_copy_is_unchanged():
    s = scored()
    v = gate.check(s, s.copy(), reps=REPS)
    assert v.outcome == "unchanged" and v.passed


def test_a_list_that_knows_the_answer_is_promoted():
    s = scored()
    y = s.not_renewed.to_numpy(float)
    v = gate.check(s, s.assign(model=y + 0.01 * s.model), reps=REPS)
    assert v.outcome == "promote"


def test_a_list_that_has_forgotten_is_blocked():
    s = scored()
    noise = rng(9).random(len(s))
    v = gate.check(s, s.assign(model=noise), reps=REPS)
    assert v.outcome == "blocked" and "behind at capacity" in v.reasons


def test_level_overall_but_worse_on_one_region_is_blocked():
    s = scored()
    y = s.not_renewed.to_numpy(bool)
    mid = (s.region == "Midwest").to_numpy()
    # Every Midwest leaver falls to the bottom; every other leaver
    # rises by enough to take the calls it frees.
    chance = s.model.to_numpy() + np.where(y & ~mid, 1.0, 0.0)
    chance[y & mid] = -1.0
    v = gate.check(s, s.assign(model=chance), reps=REPS)
    assert v.overall["precision"][0] >= 0
    assert v.outcome == "blocked"
    assert v.reasons == ["worse on region = Midwest"]


def test_a_slice_with_few_leavers_is_not_judged():
    s = scored()
    s["tenure"] = np.where(s.contract_id % 50 == 0, "under 1 year",
                           "1-3 years")
    v = gate.check(s, s.copy(), reps=REPS)
    few = v.slices[(v.slices.slice == "tenure")
                   & (v.slices.value == "under 1 year")]
    assert (few.leavers < gate.LEAST_LEAVERS).all()
    assert not few.judged.any()


class Registry:
    """Stands in for pipeline.registry.Registry: records the call."""

    def register(self, model, manifest, card=None, reason=""):
        self.reason = reason
        return 7


def test_register_refuses_what_the_gate_blocked():
    s = scored()
    blocked = gate.check(s, s.assign(model=rng(9).random(len(s))),
                         reps=REPS)
    with pytest.raises(gate.GateError, match="behind at capacity"):
        gate.register(blocked, Registry(), object(), {})
    r = Registry()
    fine = gate.check(s, s.copy(), reps=REPS)
    assert gate.register(fine, r, object(), {}) == 7
    assert r.reason.startswith("gate: unchanged")


def test_the_page_fits_the_page():
    s = scored()
    v = gate.check(s, s.assign(model=rng(9).random(len(s))), reps=REPS)
    assert max(map(len, gate.page(v, 240).splitlines())) <= 68


# ------------------------------------------------ the real cohorts
@pytest.mark.model
def test_a_tenfold_penalty_is_blocked_on_the_validation_cohorts(
        real_table):
    champion, cfg = gate.candidate()
    challenger, _ = gate.candidate(["model.strength=0.02"])
    as_of = cfg["data"]["as_of"]
    a = gate.scores(champion, real_table, as_of)
    b = gate.scores(challenger, real_table, as_of)
    v = gate.check(a, b)
    assert v.outcome == "blocked"
