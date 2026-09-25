"""
Chapter 25: a holdout that is random within each cohort, repeatable
and written once; an analysis that finds a known difference and finds
nothing where there is none; the design arithmetic of Appendix F; a
simulated world that only ever keeps called leavers; stock and queue
replays whose answers can be worked by hand; and a report that fits
the page. Made-up tables only, except the check that nothing but the
simulator reads the truth file, which reads source code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foresight.config import rng
from foresight.impact import analyse, holdout, simulate
from foresight.impact.inventory import StockCosts, cost
from foresight.impact.queue import Desk, first_read
from foresight.impact.report import render
from foresight.impact.uplift import TwoModel, features

IMPACT = Path(analyse.__file__).parent


def lists(cohorts=3, k=40):
    rows = []
    for c in range(cohorts):
        mark = pd.Timestamp("2025-01-02") + pd.DateOffset(months=c)
        for r in range(1, k + 1):
            rows.append({"moment": mark, "contract_id": 1000 * c + r,
                         "account_id": 5000 + 1000 * c + r, "rank": r,
                         "chance": 1 / (r + 1)})
    return pd.DataFrame(rows)


def records(held_rate, called_rate, cohorts=12, per_arm=20):
    """A test's record with exactly these shares leaving."""
    rows = []
    for c in range(cohorts):
        for arm, rate in (("held out", held_rate), ("called", called_rate)):
            left = np.zeros(per_arm, int)
            left[: round(rate * per_arm)] = 1
            rows += [{"moment": c, "arm": arm, "left": v} for v in left]
    return pd.DataFrame(rows)


# ------------------------------------------------ the holdout
def test_holdout_is_twenty_of_forty_in_every_cohort():
    drawn = holdout.assign(lists())
    per = drawn.groupby("moment").arm.apply(lambda a: (a == "held out")
                                            .sum())
    assert (per == holdout.HELD).all()
    assert len(drawn) == 120


def test_holdout_repeats_and_does_not_depend_on_other_cohorts():
    everything = holdout.assign(lists(3))
    alone = holdout.assign(lists(1))
    first = everything[everything.moment == everything.moment.min()]
    assert holdout.assign(lists(3)).equals(everything)
    assert first.reset_index(drop=True).arm.equals(alone.arm)


def test_cohorts_get_different_draws():
    drawn = holdout.assign(lists(2))
    a, b = (g.sort_values("rank").arm.to_numpy()
            for _, g in drawn.groupby("moment"))
    assert (a != b).any()


def test_record_writes_once_and_refuses_a_second_draw(tmp_path):
    path = tmp_path / "holdout.csv"
    drawn = holdout.assign(lists())
    assert holdout.record(drawn, path) == 120
    assert holdout.record(drawn, path) == 0
    with pytest.raises(holdout.AlreadyAssigned):
        holdout.record(holdout.assign(lists(), seed=1), path)
    assert len(holdout.load(path)) == 120


# ------------------------------------------------ the analysis
def test_compare_finds_a_known_difference():
    r = analyse.compare(records(0.30, 0.15), reps=500)
    assert r["diff"] == pytest.approx(0.15)
    assert r["save_rate"] == pytest.approx(0.5)
    assert r["lo"] < 0.15 < r["hi"]
    assert r["p_value"] < 0.01


def test_compare_finds_nothing_where_there_is_nothing():
    r = analyse.compare(records(0.25, 0.25), reps=500)
    assert r["diff"] == 0
    assert r["p_value"] == 1.0
    assert r["lo"] < 0 < r["hi"]


def test_rerandomising_keeps_each_cohorts_arms():
    rec = records(0.3, 0.1, cohorts=2)
    rec.loc[rec.moment == 1, "left"] = 0
    null = analyse.rerandomise(rec, reps=200)
    # Cohort 1 has no leavers, so only cohort 0's draw can move.
    assert np.abs(null).max() <= 0.2 + 1e-9


def test_before_after_reads_like_compare():
    before = pd.DataFrame({"left": [1, 1, 0, 0]})
    after = pd.DataFrame({"left": [1, 0, 0, 0]})
    assert analyse.before_after(before, after) == pytest.approx(0.25)


def test_sample_size_matches_appendix_f():
    assert round(analyse.sample_size(0.25, 0.1875)) == 683
    n = analyse.sample_size(0.25, 0.1875)
    assert analyse.power(0.25, 0.1875, n) == pytest.approx(0.80, abs=1e-3)
    assert analyse.cohorts_needed(0.25, 0.1875, 20, 20) == pytest.approx(
        n / 20)
    assert analyse.power(0.25, 0.25, 500) == 0.05


def test_simulated_power_is_the_size_when_calls_do_nothing():
    outcomes = (rng().random((6, 40)) < 0.25).astype(int)
    size = analyse.simulated_power(outcomes, 0.0, 12, reps=2000)
    assert 0.03 < size < 0.07
    assert analyse.simulated_power(outcomes, 0.6, 36, reps=300) > 0.9


def test_money_and_p_values_print_sensibly():
    assert analyse.dollars(-19.4) == "-$19"
    assert analyse.dollars(1234.5) == "$1,234"
    assert analyse.p_text(0.0) == "<0.0005"
    assert analyse.p_text(0.0234) == "0.023"


# ------------------------------------------------ the simulated world
def test_world_save_chances():
    flat = simulate.World(save=0.25)
    far = simulate.World(save=0.7, gone=0.6)
    p = np.array([0.0, 0.3, 0.6, 0.9])
    assert flat.save_chance(p) == pytest.approx([0.25] * 4)
    assert far.save_chance(p) == pytest.approx([0.7, 0.35, 0.0, 0.0])
    assert far.uplift(p) == pytest.approx([0.0, 0.105, 0.0, 0.0])


def test_calls_only_ever_keep_called_leavers(monkeypatch):
    rows = pd.DataFrame({"contract_id": range(200),
                         "not_renewed": [1, 0] * 100})
    monkeypatch.setattr(simulate, "true_chance",
                        lambda ids: np.full(len(ids), 0.3))
    called = np.arange(200) < 100
    world = simulate.World(save=0.5)
    left = simulate.run(rows, called, world, seed=1)
    before = rows.not_renewed.to_numpy()
    assert (left <= before).all()                 # no stayer leaves
    assert (left[~called] == before[~called]).all()
    assert 0 < simulate.saved(rows, called, left) < 50
    nothing = simulate.run(rows, called, simulate.World(save=0.0), 1)
    assert (nothing == before).all()


def test_only_the_simulator_reads_the_truth():
    for path in IMPACT.glob("*.py"):
        if path.name == "simulate.py":
            continue
        tree = ast.parse(path.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.name for n in ast.walk(tree)
                  if isinstance(n, ast.ImportFrom) for a in n.names}
        modules = {n.module for n in ast.walk(tree)
                   if isinstance(n, ast.ImportFrom)}
        assert "TRUTH" not in names, path.name
        assert "foresight.impact.simulate" not in modules, path.name
        assert "renewals.csv" not in path.read_text(), path.name


# ------------------------------------------------ stock and queue
def test_stock_cost_by_hand():
    record = pd.DataFrame({
        "series": [("Cleaning", "A", "North"), ("Cleaning", "B", "North")],
        "actual": [100.0, 50.0], "forecast": [120.0, 40.0]})
    priced = pd.DataFrame({"unit_cost": [12.0, 10.0],
                           "margin": [6.0, 4.0]}, index=["A", "B"])
    c = cost(record, "forecast", StockCosts(0.25, 1.0, 0.5), priced)
    assert c["surplus"] == pytest.approx(20 * 12 * 0.25 / 12)
    assert c["shortfall"] == pytest.approx(10 * 4 * 0.5)


def test_queue_reads_the_high_score_first_and_waits_for_opening():
    opened = pd.to_datetime(["2025-01-06 08:00", "2025-01-06 08:05",
                             "2025-01-06 08:10", "2025-01-06 20:00"])
    desk = Desk(opens=7, closes=19, minutes=60)
    fifo = first_read(opened, np.zeros(4), desk)
    urgent_last = first_read(opened, np.array([0, 0, 1, 0]), desk)
    assert fifo[:3] == pytest.approx([0, 55 / 60, 110 / 60])
    assert urgent_last[2] == pytest.approx(50 / 60)
    assert fifo[3] == pytest.approx(11.0)        # 20:00 to 07:00


# ------------------------------------------------ uplift and report
def test_two_model_uplift_finds_where_calls_help():
    g = rng()
    n = 20_000
    chance = g.uniform(0.05, 0.6, n)
    called = g.random(n) < 0.5
    helped = chance < 0.3                        # calls work only here
    p = np.where(called & helped, chance / 2, chance)
    left = (g.random(n) < p).astype(int)
    u = TwoModel().fit(features(chance), left, called).predict(
        features(chance))
    assert u[helped].mean() > u[~helped].mean()


def test_report_fits_the_page_and_says_it_is_simulated():
    f = {"year": 2025, "simulated": True, "assumed_save": 0.25,
         "retention": {"cohorts": 12, "n": 240, "held": 0.3, "diff": 0.1,
                       "lo": 0.02, "hi": 0.18, "save_rate": 0.33,
                       "save_lo": 0.07, "save_hi": 0.6,
                       "per_call": [195.0, -33.0, 424.0]},
         "design": {"cohorts": 35, "power_one_year": 0.38},
         "stock": {"series": 190, "point": 4e5, "range": 3.8e5,
                   "low": 2.2e5, "high": 7.7e5},
         "triage": {"n": 3, "waits": {"arrival": [1.0, 2.0, 3.0]},
                    "desk": "7:00 to 19:00"},
         "ask": ["Keep the holdout."]}
    page = render(f)
    assert "SIMULATED" in page
    assert max(len(line) for line in page.splitlines()) <= 68
    assert "SIMULATED" not in render(f | {"simulated": False})
