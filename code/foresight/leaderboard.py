"""
Several renewal-risk models on one page, judged the same way. Chapter 10
writes it, when Foresight first has two models to put side by side.

    python -m foresight.leaderboard          validation cohorts

evaluate.py judges one model against the rule. This module runs
evaluate() once for each model and lines the results up. Every model is
backtested on the same cohorts and bootstrapped with the same seed, so
resample i draws the same contracts for every model, and a difference
between two models on the same resample is paired, exactly as
evaluate.difference() pairs a model with the rule. run() checks that
the draws really are shared before any difference is taken.

The test year is not read here. A milestone reads it once, by name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from foresight.config import ROOT
from foresight.data.build_table import TABLE
from foresight.evaluate import REPS, SPLITS, evaluate, interval
from foresight.models.forest import RenewalForest
from foresight.models.logistic import CALLS, RenewalRisk

MODELS = {"logistic": RenewalRisk, "forest": RenewalForest}
BOARD = ROOT / "docs" / "foresight-leaderboard.md"
SHOWN = [("precision", "pct"), ("auc", "num")]  # measure, format


def run(models: dict | None = None, split: str = "validation",
        table=None, k: int = CALLS, reps: int = REPS) -> dict:
    """evaluate() for every model, on the same cohorts and resamples."""
    models = MODELS if models is None else models
    table = pd.read_parquet(TABLE) if table is None else table
    runs = {name: evaluate(split, table, k, reps, make)
            for name, make in models.items()}
    first = next(iter(runs.values()))
    for ev in runs.values():            # the rule's draws must agree
        if not ev["draws"].filter(like="rule").equals(
                first["draws"].filter(like="rule")):
            raise ValueError("models were not scored on the same "
                             "resamples; differences would not pair")
    return {"split": split, "k": k, "runs": runs}


def point(board: dict, what: str, name: str) -> float:
    """A model's, the rule's or a baseline's value on the real rows."""
    runs = board["runs"]
    ev = runs[name] if name in runs else next(iter(runs.values()))
    return ev["point"][what]["model" if name in runs else name]


def draws(board: dict, what: str, name: str) -> pd.Series:
    """The same value on every resample."""
    runs = board["runs"]
    ev = runs[name] if name in runs else next(iter(runs.values()))
    return ev["draws"][(what, "model" if name in runs else name)]


def paired(board: dict, what: str, a: str, b: str) -> tuple:
    """a minus b, and the 95% interval of the paired difference."""
    d = draws(board, what, a) - draws(board, what, b)
    return (point(board, what, a) - point(board, what, b),
            *interval(d))


def _cell(v: float, lo: float, hi: float, kind: str) -> str:
    if kind == "pct":
        return f"{v * 100:.1f} ({lo * 100:.1f}-{hi * 100:.1f})"
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})"


def _diff(v: float, lo: float, hi: float, kind: str) -> str:
    if kind == "pct":
        return f"{v * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f})"
    return f"{v:+.3f} ({lo:+.3f} to {hi:+.3f})"


def page(board: dict) -> str:
    """The leaderboard as fixed-width text no wider than 68."""
    k, runs = board["k"], board["runs"]
    names = list(runs)
    s = runs[names[0]]["scored"]
    first, last = SPLITS[board["split"]]
    calls = int(s.groupby("moment").size().clip(upper=k).sum())
    out = ["Foresight leaderboard: renewal risk", "",
           f"Split      {board['split']}: contracts ending {first}"
           f" to {last}",
           f"{'':11}{s.moment.nunique()} monthly cohorts, {len(s):,}"
           f" contracts, {int(s.not_renewed.sum())} leavers",
           f"{'':11}{calls} calls: the top {k} of each cohort",
           "Method     every model by foresight.evaluate's backtest,",
           f"{'':11}fitted on the outcomes known at each cohort's mark",
           f"Intervals  95%, {len(runs[names[0]]['draws']):,}"
           " bootstrap resamples, the same for every list", "",
           f"{'':10}{'leavers':>8}  {'precision at ' + str(k):<18}AUC"]
    for name in names + ["rule"]:
        cells = [_cell(point(board, w, name),
                       *interval(draws(board, w, name)), kind)
                 for w, kind in SHOWN]
        hits = round(point(board, "precision", name) * calls)
        out.append(f"{name:<10}{hits:>8}  {cells[0]:<18}{cells[1]}")
    for name in ("random", "perfect"):
        v = point(board, "precision", name)
        auc = "0.500" if name == "random" else "1.000"
        out.append(f"{name:<10}{v * calls:>8.0f}  {v * 100:<18.1f}"
                   f"{auc}")
    out += ["", "Paired differences (95%)",
            f"{'':20}{'precision, points':<22}AUC"]
    pairs = [(n, "rule") for n in names]
    pairs += [(n, names[0]) for n in names[1:]]
    for a, b in pairs:
        cells = [_diff(*paired(board, w, a, b), kind)
                 for w, kind in SHOWN]
        out.append(f"{a + ' - ' + b:<20}{cells[0]:<22}{cells[1]}")
    out += ["", "A difference whose interval includes zero is not a",
            "difference these cohorts can see."]
    return "\n".join(out)


def report(board: dict, path: Path = BOARD) -> str:
    """Write the page to `path` as Markdown, and return its text."""
    text = page(board)
    head, _, body = text.partition("\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {head}\n\n```text\n{body}\n```\n")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=BOARD)
    args = ap.parse_args(argv)
    print(report(run(), args.out))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
