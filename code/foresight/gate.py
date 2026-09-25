"""
The gate a retrained model must pass before it is registered.
Chapter 23 writes it.

    python -m foresight.gate scores OUT.csv [--set section.key=value]
    python -m foresight.gate compare CHAMPION.csv CHALLENGER.csv

scores() backtests one model on the latest cohorts whose outcomes
were known on the settings' as_of day, each cohort scored by a fresh
copy fitted only on what was on record at its mark, and labels every
contract with Chapter 16's slices. CI runs it twice, once with the
code on main (the champion) and once with the code on the branch
(the challenger), so a change to the code and a change to the
settings are judged the same way.

check() is Chapter 24's rule (monitor.policy.gate) on those cohorts,
with one condition added for the slices:

    blocked     behind at capacity, or clearly worse on a protected
                slice: the paired interval for the share of that
                slice's leavers the list calls lies wholly below zero
    promote     not blocked, and a paired interval on precision at
                capacity or on AUC within lists clears zero in the
                challenger's favour: Chapter 24 would promote it
    register    not blocked, not better: it may be registered as
                staged, and production stays as it is
    unchanged   every chance equal to the champion's

A slice with fewer than LEAST_LEAVERS leavers is reported and not
judged: its interval runs from nothing to everything. register()
refuses a blocked model, so it never reaches the registry.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from foresight import slices
from foresight.config import SEED, rng
from foresight.decide import by_capacity
from foresight.evaluate import (HISTORY_FROM, REPS, backtest, interval,
                                known_by)
from foresight.models.logistic import CALLS
from foresight.monitor import policy

PROTECTED = slices.SLICES       # region, segment, size, tenure, ...
LEAST_LEAVERS = 10              # fewer, and a slice is not judged
SAME = 1e-9                     # chances closer than this are equal
TRAINING = ("2023-01-01", "2024-06-30")   # where slices are cut
KEEP = ["moment", "contract_id", "account_id", "end_date",
        "not_renewed", "model", "rule", *PROTECTED]


class GateError(Exception):
    """A model the gate blocked was about to be registered."""


# ------------------------------------------------ the evidence
def scores(model, table: pd.DataFrame, as_of: str,
           cohorts: int = 6, keys: set | None = None) -> pd.DataFrame:
    """Backtest `model` on the latest `cohorts` cohorts whose outcomes
    were on record on `as_of`, with every slice labelled."""
    from foresight.pipeline.model import maker
    history = table[table.end_date >= HISTORY_FROM]
    rows = known_by(history, as_of)
    ends = sorted(rows.end_date.unique())[-cohorts:]
    first = str(ends[0].replace(day=1).date())
    scored = backtest(rows, first, str(ends[-1].date()), maker(model))
    train = table[table.end_date.between(*TRAINING)]
    keys = slices.key_accounts() if keys is None else keys
    return slices.label(scored, train, keys)[KEEP]


def _shadow(champion: pd.DataFrame, challenger: pd.DataFrame
            ) -> pd.DataFrame:
    """Both scored frames in the shape policy.gate() reads."""
    cols = ["moment", "contract_id", "end_date", "not_renewed", "rule"]
    return pd.concat([
        champion[cols].assign(name="champion", chance=champion.model),
        challenger[cols].assign(name="challenger",
                                chance=challenger.model)],
        ignore_index=True)


def _slice_hits(code, y, called, n):
    """Leavers called, per value of a slice."""
    return np.bincount(code, weights=y * called, minlength=n)


def by_slice(champion: pd.DataFrame, challenger: pd.DataFrame,
             by: str, reps: int = REPS, seed: int = SEED
             ) -> pd.DataFrame:
    """For each value of `by`: its leavers, how many of them each list
    called, and challenger minus champion as a share of the leavers,
    with a 95% interval from resampling contracts within each cohort.
    The calls are fixed, as in Chapter 16's slice table."""
    cats = pd.Categorical(champion[by])
    code, names = cats.codes, list(cats.categories)
    n = len(names)
    y = champion.not_renewed.to_numpy(dtype=float)
    a = by_capacity(challenger).astype(float)
    b = by_capacity(champion).astype(float)
    leavers = np.bincount(code, weights=y, minlength=n)
    ha, hb = _slice_hits(code, y, a, n), _slice_hits(code, y, b, n)
    g, draws = rng(seed), []
    groups = list(champion.groupby("moment").indices.values())
    for _ in range(reps):
        i = np.concatenate([r[g.integers(0, len(r), len(r))]
                            for r in groups])
        left = np.bincount(code[i], weights=y[i], minlength=n)
        diff = (_slice_hits(code[i], y[i], a[i], n)
                - _slice_hits(code[i], y[i], b[i], n))
        with np.errstate(invalid="ignore", divide="ignore"):
            draws.append(diff / left)
    d = np.array(draws)
    out = []
    for j, name in enumerate(names):
        vals = d[:, j][~np.isnan(d[:, j])]
        lo, hi = interval(vals) if len(vals) else (np.nan, np.nan)
        out.append({"slice": by, "value": str(name),
                    "leavers": int(leavers[j]),
                    "champion": int(hb[j]), "challenger": int(ha[j]),
                    "diff": ((ha[j] - hb[j]) / leavers[j]
                             if leavers[j] else np.nan),
                    "lo": lo, "hi": hi})
    return pd.DataFrame(out)


# ------------------------------------------------ the rule
@dataclass
class Verdict:
    outcome: str                # blocked, promote, register, unchanged
    overall: dict               # policy.gate()'s result
    slices: pd.DataFrame        # by_slice() for every protected slice
    reasons: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.outcome != "blocked"


def check(champion: pd.DataFrame, challenger: pd.DataFrame,
          protected=PROTECTED, reps: int = REPS) -> Verdict:
    """Chapter 24's rule on the overall list, and no protected slice
    clearly worse."""
    assert (champion.contract_id.to_numpy()
            == challenger.contract_id.to_numpy()).all()
    day = champion.end_date.max() + pd.Timedelta(days=1)
    overall = policy.gate(_shadow(champion, challenger), "champion",
                          "challenger", day)
    table = pd.concat([by_slice(champion, challenger, by, reps)
                       for by in protected], ignore_index=True)
    table["judged"] = table.leavers >= LEAST_LEAVERS
    table["worse"] = table.judged & (table.hi < 0)
    same = np.abs(champion.model.to_numpy()
                  - challenger.model.to_numpy()).max() < SAME
    reasons = []
    if same:
        return Verdict("unchanged", overall, table,
                       ["every chance equals the champion's"])
    if "precision" not in overall:
        reasons.append(f"only {overall['cohorts']} cohorts with"
                       " outcomes; the gate needs"
                       f" {policy.LEAST}")
    elif overall["precision"][0] < 0:
        reasons.append("behind at capacity")
    for r in table[table.worse].itertuples():
        reasons.append(f"worse on {r.slice} = {r.value}")
    if reasons:
        return Verdict("blocked", overall, table, reasons)
    if overall["promote"]:
        return Verdict("promote", overall, table,
                       ["ahead, with an interval clear of zero"])
    return Verdict("register", overall, table,
                   ["not behind, and not shown to be better"])


def register(verdict: Verdict, registry, model, manifest: dict,
             card: str | None = None) -> int:
    """Register `model` as staged, if the gate let it through."""
    if not verdict.passed:
        raise GateError("the gate blocked this model: "
                        + "; ".join(verdict.reasons))
    reason = f"gate: {verdict.outcome}; " + "; ".join(verdict.reasons)
    return registry.register(model, manifest, card, reason=reason)


# ------------------------------------------------ the page
def _pts(v: float) -> str:
    s = f"{v * 100:+.1f}"
    return " 0.0" if s in ("+0.0", "-0.0") else s


def page(v: Verdict, calls: int) -> str:
    """The verdict, the overall differences and every judged slice
    that moved, in lines no wider than 68."""
    out = [f"gate: {v.outcome.upper()}"]
    out += [f"  - {r}" for r in v.reasons]
    if "precision" in v.overall:
        p, a = v.overall["precision"], v.overall["auc"]
        out.append(f"  leavers at capacity {p[0] * calls:+.0f}"
                   f" ({p[1] * calls:+.1f} to {p[2] * calls:+.1f})"
                   f" in {calls} calls")
        out.append(f"  AUC within lists    {a[0]:+.4f}"
                   f" ({a[1]:+.4f} to {a[2]:+.4f})")
    moved = v.slices[v.slices.judged
                     & (v.slices.champion != v.slices.challenger)]
    if len(moved):
        out.append(f"  {'slice':<22}{'leavers':>8}{'called':>11}"
                   f"{'change':>7}{'(95%)':>16}")
    for r in moved.itertuples():
        ci = f"{_pts(r.lo)} to {_pts(r.hi)}"
        out.append(f"  {r.slice + ' ' + r.value:<22}{r.leavers:>8}"
                   f"{r.champion:>5} ->{r.challenger:>3}"
                   f"{_pts(r.diff):>7}{ci:>16}"
                   + (" *" if r.worse else ""))
    judged = int(v.slices.judged.sum())
    out.append(f"  {judged} slices judged; {len(v.slices) - judged}"
               f" have under {LEAST_LEAVERS} leavers."
               + (" * worse: all below zero" if v.slices.worse.any()
                  else ""))
    return "\n".join(out)


# ------------------------------------------------ the command
def candidate(overrides=()):
    """The renewal model the settings file describes, with any
    overrides, and the settings themselves."""
    from foresight.pipeline import settings
    from foresight.pipeline.model import build
    cfg = settings.load("renewal", overrides)
    m = cfg["model"]
    return build(m["strength"], m["calibration_months"],
                 m["no_order_days"]), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scores", help="backtest this code's model")
    s.add_argument("out", type=Path)
    s.add_argument("--set", dest="overrides", action="append",
                   default=[], metavar="SECTION.KEY=VALUE")
    c = sub.add_parser("compare", help="champion against challenger")
    c.add_argument("champion", type=Path)
    c.add_argument("challenger", type=Path)
    args = ap.parse_args(argv)
    if args.command == "scores":
        from foresight.pipeline import features
        model, cfg = candidate(args.overrides)
        d = cfg["data"]
        table = features.labelled(d["first"], d["last"])
        out = scores(model, table, d["as_of"],
                     cfg["evaluation"]["cohorts"])
        out.to_csv(args.out, index=False)
        print(f"{len(out):,} contracts in {out.moment.nunique()}"
              f" cohorts -> {args.out}")
        return 0
    read = {"parse_dates": ["moment", "end_date"]}
    champ = pd.read_csv(args.champion, **read)
    chall = pd.read_csv(args.challenger, **read)
    v = check(champ, chall)
    print(page(v, CALLS * champ.moment.nunique()))
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
