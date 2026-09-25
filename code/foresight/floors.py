"""
The least a renewal model may score, overall and in each slice, and
where those numbers come from. Chapter 23 writes it.

A floor is the lower end of the 95% interval that the reference model
(v0.6) earned in its own backtest on the same cohorts: the worst it
could plausibly have scored on another draw of the same kind of
contracts. A model below a floor is worse than the reference by more
than luck would explain. The floors are set once, from a printed
backtest (listing 23/05), and written to tests/model_floors.json,
where the model tests read them; they move only when a person decides
the reference has changed.

    overall     precision at capacity and AUC (Chapter 8's bootstrap);
                and the list must not trail Chapter 3's rule
    per slice   the share of the slice's leavers the list calls
                (Chapter 16's slice table, one minus its miss rate),
                for every slice with at least LEAST_LEAVERS leavers
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from foresight import slices
from foresight.config import ROOT
from foresight.decide import by_capacity
from foresight.evaluate import REPS, bootstrap, interval, measure
from foresight.gate import LEAST_LEAVERS, PROTECTED

FILE = ROOT / "tests" / "model_floors.json"


def _down(x: float) -> float:
    """Rounded down to four places, so the reference always passes."""
    return math.floor(x * 10_000) / 10_000


def called_share(scored: pd.DataFrame, by: str) -> pd.Series:
    """For each value of `by`, the share of its leavers the list
    called."""
    called = by_capacity(scored)
    y = scored.not_renewed.astype(bool)
    hit = (called & y).groupby(scored[by], observed=True).sum()
    return hit / y.groupby(scored[by], observed=True).sum()


def set_floors(scored: pd.DataFrame, reps: int = REPS) -> dict:
    """Floors from the reference model's backtest."""
    draws = bootstrap(scored.assign(base=0.0), reps=reps)
    out = {"precision": _down(interval(draws["precision",
                                             "model"])[0]),
           "auc": _down(interval(draws["auc", "model"])[0]),
           "slices": {}}
    for by in PROTECTED:
        t = slices.table(scored, by, by_capacity(scored), reps)
        kept = t[t.leavers >= LEAST_LEAVERS]
        out["slices"][by] = {str(name): _down(1 - r.miss_hi)
                             for name, r in kept.iterrows()}
    return out


def shortfalls(scored: pd.DataFrame, floors: dict) -> list[str]:
    """Every floor the scored list falls below."""
    m, out = measure(scored.assign(base=0.0)), []
    if m["precision"]["model"] < m["precision"]["rule"]:
        out.append("precision at capacity below the rule's")
    for what in ("precision", "auc"):
        if m[what]["model"] < floors[what]:
            out.append(f"{what} {m[what]['model']:.4f} below"
                       f" {floors[what]:.4f}")
    for by, floor in floors["slices"].items():
        got = called_share(scored, by)
        for name, least in floor.items():
            if got.get(name, 0.0) < least:
                out.append(f"{by} {name}: {got.get(name, 0.0):.3f}"
                           f" of leavers called, below {least:.4f}")
    return out


def write(floors: dict, about: dict, path: Path = FILE) -> None:
    path.write_text(json.dumps({**about, **floors}, indent=1) + "\n")


def read(path: Path = FILE) -> dict:
    return json.loads(path.read_text())
