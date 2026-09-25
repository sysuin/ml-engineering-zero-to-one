"""
A stand-in for the world, for Chapter 25's retention test. Not part of
what Meridian would run.

Meridian's warehouse records no retention calls, so there is no real
experiment to analyse. Chapter 25 runs the programme in a simulation
instead, and this module plays the part of the accounts: it decides
which called would-be leavers a call keeps. Everything it decides rests
on an assumption, and every assumption is an argument you can change.

This is the only module in `foresight.impact` that reads the
generator's truth file (`truth/renewals.csv`, each contract's true
chance of leaving). The world is allowed to know it; the analysis is
not. `analyse.py`, `holdout.py` and `report.py` see only what Meridian
would record: who was on the list, who was held out, who renewed.

What a call does, in a World:

    save    the chance a call keeps a would-be leaver who was barely
            leaving (true chance near zero). The brief's one in four.
    gone    the true chance of leaving at and above which an account
            has decided, and no call changes its mind. Between 0 and
            `gone` the save chance falls in a straight line from `save`
            to zero. None means it never falls: every would-be leaver
            is saved with chance `save`, as the brief assumes.

A call never makes a stayer leave in this world. Real calls can; the
chapter says so and does not simulate it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import TRUTH

TRUTH_FILE = TRUTH / "renewals.csv"


@dataclass(frozen=True)
class World:
    save: float = 0.25          # the brief's assumption (Chapter 3)
    gone: float | None = None   # None: every leaver equally persuadable

    def save_chance(self, p_leave) -> np.ndarray:
        """The chance a call keeps a would-be leaver, by its true
        chance of leaving."""
        p = np.asarray(p_leave, dtype=float)
        if self.gone is None:
            return np.full(p.shape, self.save)
        return self.save * np.clip(1 - p / self.gone, 0, 1)

    def uplift(self, p_leave) -> np.ndarray:
        """How much a call lowers the chance of leaving: the chance the
        account was leaving, times the chance a call keeps it."""
        p = np.asarray(p_leave, dtype=float)
        return p * self.save_chance(p)


@lru_cache(maxsize=1)
def _truth(path: Path = TRUTH_FILE) -> pd.Series:
    t = pd.read_csv(path, usecols=["contract_id", "p_leave"])
    return t.set_index("contract_id").p_leave


def true_chance(contract_ids) -> np.ndarray:
    """Each contract's true chance of leaving, from the generator."""
    return _truth().reindex(np.asarray(contract_ids)).to_numpy()


def run(rows: pd.DataFrame, called, world: World,
        seed: int) -> np.ndarray:
    """What the warehouse would record after the calls: `rows` carry
    the recorded outcome (`not_renewed`), which is what happened with
    nobody called. A called contract that left is kept with the World's
    save chance. Returns the new outcomes, 1 for left."""
    left = rows.not_renewed.to_numpy().astype(int).copy()
    called = np.asarray(called, dtype=bool)
    s = world.save_chance(true_chance(rows.contract_id))
    kept = np.random.default_rng(seed).random(len(rows)) < s
    left[called & (left == 1) & kept] = 0
    return left


def saved(rows: pd.DataFrame, called, after) -> int:
    """Leavers the calls kept: known here, and nowhere else."""
    before = rows.not_renewed.to_numpy().astype(int)
    return int(((before == 1) & (np.asarray(after) == 0)
                & np.asarray(called, dtype=bool)).sum())


def fresh(p_leave, called, world: World, g) -> np.ndarray:
    """A whole new year of outcomes for the same contracts: who would
    have left, drawn from the true chances, then the calls. For power
    and for the persuadables, where one recorded year is too few."""
    p = np.asarray(p_leave, dtype=float)
    left = g.random(p.shape) < p
    kept = g.random(p.shape) < world.save_chance(p)
    return (left & ~(np.asarray(called, dtype=bool) & kept)).astype(int)
