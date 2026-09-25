"""
Foresight's evaluation. Every renewal-risk model from v0.3 on is judged
here, the same way. Chapter 8 writes it.

    python -m foresight.evaluate                  validation cohorts
    python -m foresight.evaluate --split test     a milestone's one look

A rolling-origin backtest: for each monthly cohort in the split, the
model is fitted on the contracts whose outcome was on record on that
cohort's mark, the morning the list is made, and on nothing later. The
rule and a random list are scored on the same rows every time. Every
number has a bootstrap interval, and the model is compared with the
rule by the paired difference, never by two separate intervals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from foresight.config import ROOT, SEED, rng
from foresight.data.build_table import TABLE
from foresight.models.logistic import (CALLS, RenewalRisk,
                                       days_since_rule, log_loss)

SPLITS = {"validation": ("2024-07-01", "2024-12-31"),
          "test": ("2025-01-01", "2025-12-31")}
HISTORY_FROM = "2023-01-01"     # contracts ending in 2022: not used
REPS = 2000                     # bootstrap resamples
REPORT = ROOT / "docs" / "foresight-evaluation.md"


# ------------------------------------------------ the backtest
def known_by(table: pd.DataFrame, day) -> pd.DataFrame:
    """Rows whose outcome was on record on `day`: the contract had
    ended. build_table's known_by applies the same rule."""
    return table[table.end_date < pd.Timestamp(day)]


def backtest(table: pd.DataFrame, first: str, last: str,
             make_model=RenewalRisk) -> pd.DataFrame:
    """Score every cohort ending in [first, last] with a model fitted
    on the outcomes known at that cohort's mark."""
    history = table[table.end_date >= HISTORY_FROM]
    rows = table[table.end_date.between(first, last)]
    scored = []
    for mark, cohort in rows.groupby("moment"):
        train = known_by(history, mark)
        model = make_model().fit(train)
        scored.append(cohort.assign(
            model=model.predict_proba(cohort),
            rule=days_since_rule(cohort),
            base=train.not_renewed.mean(),     # the base-rate list
            trained_on=len(train)))
    return pd.concat(scored).reset_index(drop=True)


# ------------------------------------------------ the measures
def auc(y, score) -> float:
    """The chance that a random leaver outscores a random renewer,
    a tie counting as half."""
    y = np.asarray(y)
    ranks = rankdata(score)                 # ties share their rank
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    """The mean squared gap between the probability and the outcome."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def hits_at_k(y, score, ids, k: int = CALLS) -> int:
    """Leavers among the k highest scores; a tie goes to the lower id,
    as in Chapter 7's top_of_each_cohort()."""
    top = np.lexsort((ids, -np.asarray(score, dtype=float)))[:k]
    return int(np.asarray(y)[top].sum())


def measure(scored: pd.DataFrame, k: int = CALLS) -> dict:
    """Every headline number, for the model and its baselines."""
    y = scored.not_renewed.to_numpy()
    ids = scored.contract_id.to_numpy()
    lists = {n: scored[n].to_numpy(dtype=float)
             for n in ("model", "rule")}
    hits = {"model": 0, "rule": 0, "random": 0.0, "perfect": 0}
    calls = 0
    for rows in scored.groupby("moment").indices.values():
        cy, n = y[rows], min(k, len(rows))
        calls += n
        for name, score in lists.items():
            hits[name] += hits_at_k(cy, score[rows], ids[rows], k)
        hits["random"] += n * float(cy.mean())      # expected
        hits["perfect"] += min(n, int(cy.sum()))    # leavers first
    out = {"precision": {n: h / calls for n, h in hits.items()},
           "recall": {n: h / int(y.sum()) for n, h in hits.items()}}
    out["auc"] = {"model": auc(y, lists["model"]),
                  "rule": auc(y, lists["rule"]), "random": 0.5}
    p, base = lists["model"], scored.base.to_numpy()
    out["log loss"] = {"model": log_loss(y, p),
                       "base rate": log_loss(y, base)}
    out["brier"] = {"model": brier(y, p), "base rate": brier(y, base)}
    return out


# ------------------------------------------------ how sure
def resample(scored: pd.DataFrame, g) -> pd.DataFrame:
    """One bootstrap sample: contracts drawn with replacement within
    each cohort, so every cohort keeps its size. No account appears
    twice in a split, so this is also a resample of accounts."""
    draw = [rows[g.integers(0, len(rows), len(rows))]
            for rows in scored.groupby("moment").indices.values()]
    return scored.take(np.concatenate(draw))


def bootstrap(scored: pd.DataFrame, k: int = CALLS, reps: int = REPS,
              seed: int = SEED) -> pd.DataFrame:
    """measure() on `reps` resamples: one row per resample, one column
    per (measure, list)."""
    g = rng(seed)
    scored = scored[["moment", "contract_id", "not_renewed", "model",
                     "rule", "base"]]          # all measure() reads
    rows = []
    for _ in range(reps):
        m = measure(resample(scored, g), k)
        rows.append({(a, b): v for a, d in m.items()
                     for b, v in d.items()})
    return pd.DataFrame(rows)


def interval(values, level: float = 0.95) -> tuple[float, float]:
    """The middle `level` of the resampled values."""
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(lo), float(hi)


def difference(draws: pd.DataFrame, what: str, a: str = "model",
               b: str = "rule") -> pd.Series:
    """a minus b on each resample: the paired difference."""
    return draws[(what, a)] - draws[(what, b)]


def evaluate(split: str = "validation", table=None, k: int = CALLS,
             reps: int = REPS, make_model=RenewalRisk) -> dict:
    """Backtest, measure, bootstrap: everything the report needs."""
    table = pd.read_parquet(TABLE) if table is None else table
    scored = backtest(table, *SPLITS[split], make_model)
    return {"split": split, "k": k, "scored": scored,
            "point": measure(scored, k),
            "draws": bootstrap(scored, k, reps)}


# ------------------------------------------------ the page
ROWS = [("precision", "precision at {k}", "pct"),
        ("recall", "recall at {k}", "pct"),
        ("auc", "AUC", "num"),
        ("log loss", "log loss", "num"),
        ("brier", "Brier score", "num")]


def _fmt(v: float, kind: str) -> str:
    return f"{v * 100:.1f}" if kind == "pct" else f"{v:.3f}"


def _with_interval(ev: dict, what: str, name: str, kind: str) -> str:
    """'23.8 (18.3-29.6)': the value and its 95% interval."""
    if name not in ev["point"][what]:
        return "-"
    v = _fmt(ev["point"][what][name], kind)
    if name in ("random", "perfect", "base rate"):
        return v                        # a baseline: shown bare
    lo, hi = interval(ev["draws"][(what, name)])
    return f"{v} ({_fmt(lo, kind)}-{_fmt(hi, kind)})"


def _difference(ev: dict, what: str, kind: str) -> str:
    d = ev["point"][what]["model"] - ev["point"][what]["rule"]
    lo, hi = interval(difference(ev["draws"], what))
    if kind == "pct":
        d, lo, hi = d * 100, lo * 100, hi * 100
        return f"{d:+.1f} points ({lo:+.1f} to {hi:+.1f})"
    return f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"


def verdict(ev: dict) -> str:
    """The headline, computed rather than written."""
    lo, hi = interval(difference(ev["draws"], "precision"))
    if lo > 0:
        return "The model's list beats the rule's at capacity."
    if hi < 0:
        return "The rule's list beats the model's at capacity."
    return "Model and rule cannot be told apart at capacity."


def page(ev: dict, title: str = "Foresight evaluation: renewal risk"
         ) -> str:
    """The one-page report, as fixed-width text no wider than 68."""
    s, k = ev["scored"], ev["k"]
    first, last = SPLITS[ev["split"]]
    fewest, most = s.trained_on.min(), s.trained_on.max()
    pad = " " * 11
    out = [title, "",
           f"Split      {ev['split']}: contracts ending {first}"
           f" to {last}",
           f"{pad}{s.moment.nunique()} monthly cohorts, {len(s):,}"
           f" contracts, {int(s.not_renewed.sum())} leavers",
           "Method     rolling-origin backtest: each cohort scored"
           " by a",
           f"{pad}model fitted on the {fewest:,} to {most:,} outcomes",
           f"{pad}known at its mark; rule and random on the same rows",
           f"Intervals  95%, {len(ev['draws']):,} bootstrap resamples"
           " of contracts",
           f"{pad}within each cohort", "",
           f"Verdict    {verdict(ev)}", "",
           f"{'':16}{'model':<21}{'rule':<21}{'baseline':>10}"]
    for what, name, kind in ROWS:
        base = "random" if what in ("precision", "recall", "auc") \
            else "base rate"
        out.append(f"{name.format(k=k):<16}"
                   f"{_with_interval(ev, what, 'model', kind):<21}"
                   f"{_with_interval(ev, what, 'rule', kind):<21}"
                   f"{_with_interval(ev, what, base, kind):>10}")
    out += ["", "Precision and recall in per cent. Baseline: a random",
            "list; for log loss and Brier, the training base rate for",
            "every contract (the rule gives no probability to score).",
            "", "Model minus rule, paired"]
    for what, name, kind in ROWS[:3]:
        out.append(f"  {name.format(k=k):<18}"
                   f"{_difference(ev, what, kind)}")
    out += ["", f"{'cohort mark':<14}{'contracts':>10}{'leavers':>9}"
            f"{'model hits':>12}{'rule hits':>11}"]
    total = {"model": 0, "rule": 0}
    for mark, c in s.groupby("moment"):
        y, ids = c.not_renewed.to_numpy(), c.contract_id.to_numpy()
        hits = {n: hits_at_k(y, c[n].to_numpy(), ids, k) for n in total}
        total = {n: total[n] + hits[n] for n in total}
        out.append(f"{mark:%Y-%m-%d}{len(c):>14}{int(y.sum()):>9}"
                   f"{hits['model']:>12}{hits['rule']:>11}")
    out.append(f"{'all':<10}{len(s):>14,}{int(s.not_renewed.sum()):>9}"
               f"{total['model']:>12}{total['rule']:>11}")
    perfect = ev["point"]["precision"]["perfect"]
    out += ["", f"A perfect list would reach {perfect:.1%} precision.",
            "Not measured here: calibration and value in dollars",
            "(Chapter 14), performance by segment (Chapter 16)."]
    return "\n".join(out)


def report(ev: dict, path: Path = REPORT, title: str = "") -> str:
    """Write the page to `path` as Markdown, and return its text."""
    text = page(ev, title) if title else page(ev)
    head, _, body = text.partition("\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {head}\n\n```text\n{body}\n```\n")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=list(SPLITS),
                    default="validation")
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    print(report(evaluate(args.split), args.out))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
