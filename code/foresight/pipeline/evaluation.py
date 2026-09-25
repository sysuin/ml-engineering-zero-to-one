"""
What the training job measures, and the model card it fills in.
Chapter 21 writes it.

scores() takes the backtest the leakage checks already ran, the latest
cohorts whose outcomes the job may see, and gives the numbers every
run records: leavers in the top 40s, precision, AUC and log loss, the
rule beside each, and the calibration slope. card() gathers what
Chapter 16's template asks for, measured on the same cohorts, and
foresight.card.render() writes it. Chapter 16's listing 14 gathered
the same numbers for the validation cohorts; for a job trained on 1
January 2025 the cohorts are the validation cohorts, and the card is
Chapter 16's, line for line, until the test year's section, which the
training job never writes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from foresight.decide import COSTS, by_capacity, calibration_slope
from foresight.evaluate import (HISTORY_FROM, REPS, backtest, bootstrap,
                                interval, known_by, measure)
from foresight.explain import RARE, SMALLEST
from foresight.pipeline.model import maker, weights
from foresight.slices import key_accounts, label, table

HALLOWAY = (8118, "2024-04-01")     # its last contract, and its mark


def scores(scored: pd.DataFrame, reps: int = REPS) -> dict:
    """The run's metrics on its backtest, with 95% intervals."""
    pt, dr = measure(scored), bootstrap(scored, reps=reps)
    calls = int(scored.groupby("moment").size().clip(upper=40).sum())
    out = {"cohorts": scored.moment.nunique(), "contracts": len(scored),
           "leavers": int(scored.not_renewed.sum()), "calls": calls}
    for name in ("model", "rule"):
        out[f"{name} hits"] = round(pt["precision"][name] * calls)
        for what in ("precision", "auc"):
            lo, hi = interval(dr[(what, name)])
            out[f"{name} {what}"] = pt[what][name]
            out[f"{name} {what} lo"], out[f"{name} {what} hi"] = lo, hi
    out["log loss"] = pt["log loss"]["model"]
    out["base log loss"] = pt["log loss"]["base rate"]
    out["slope"] = calibration_slope(scored.model,
                                     scored.not_renewed)[0]
    return out


def _cell(m: dict, what: str, pct: bool) -> str:
    v, lo, hi = (m[f"{what}{s}"] for s in ("", " lo", " hi"))
    if pct:
        return f"{v:.1%} ({lo:.1%}-{hi:.1%})"
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})"


def _pts(v: float) -> str:
    v = f"{v * 100:+.1f}"
    return "0.0" if v in ("+0.0", "-0.0") else v


def card(model, table_: pd.DataFrame, scored: pd.DataFrame, m: dict,
         settings: dict) -> dict:
    """Every number Chapter 16's card template asks for."""
    cfg, ceiling = settings["model"], settings["evaluation"]["ceiling"]
    first = str(scored.end_date.min().replace(day=1).date())
    last = str(scored.end_date.max().date())
    measured = (ceiling["first"], ceiling["last"]) == (first, last)
    history = table_[table_.end_date >= HISTORY_FROM]
    lasso = model.estimator                   # the lasso, uncalibrated
    raw = backtest(table_, first, last, maker(lasso))
    raw_slope = calibration_slope(raw.model, raw.not_renewed)[0]

    s = label(scored, history[history.end_date < first], key_accounts())
    called = by_capacity(s)
    lines = ["| Slice | Contracts | Leavers | Missed (95%) | Renewers"
             " called | Chance minus share left, points (95%) |",
             "|---|---|---|---|---|---|"]
    t = {}
    for by in ("segment", "size", "population"):
        t[by] = table(s, by, called)
        for name, r in t[by].iterrows():
            miss = ("" if r.leavers == 0 else
                    f"{r.miss:.0%} ({r.miss_lo:.0%} to"
                    f" {r.miss_hi:.0%})")
            lines.append(f"| {by}: {name} | {r.contracts:,.0f} |"
                         f" {r.leavers:.0f} | {miss} | {r.fpr:.0%} |"
                         f" {_pts(r.gap)} ({_pts(r.gap_lo)} to"
                         f" {_pts(r.gap_hi)}) |")

    keys = key_accounts()
    kh = history[history.end_date.between(HISTORY_FROM, last)]
    kh = kh[kh.account_id.isin(keys)]
    contract, mark = HALLOWAY
    halloway = _fit(model, known_by(history, mark)).predict_proba(
        table_[table_.contract_id == contract])[0, 1]
    marks = np.sort(scored.moment.unique())[[0, -1]]
    w = [weights(_fit(model, known_by(history, d)))
         ["orders_prev_90d"] for d in marks]
    seg, size = t["segment"], t["size"]
    return {
        "version": cfg["version"], "calls": 40,
        "break_even": f"{COSTS.break_even():.1%}",
        "key_contracts": len(kh), "key_left": int(kh.not_renewed.sum()),
        "halloway": f"{halloway:.2g}", "strength": cfg["strength"],
        "cohorts": m["cohorts"], "first": first, "last": last,
        "contracts": f"{m['contracts']:,}", "leavers": m["leavers"],
        "calls_total": m["calls"], "reps": f"{REPS:,}",
        "hits": m["model hits"], "rule_hits": m["rule hits"],
        "ceiling_hits": ceiling["hits"] if measured else "not measured",
        "precision": _cell(m, "model precision", True),
        "rule_precision": _cell(m, "rule precision", True),
        "ceiling_precision": ceiling["precision"] if measured else "",
        "auc": _cell(m, "model auc", False),
        "rule_auc": _cell(m, "rule auc", False),
        "ceiling_auc": ceiling["auc"] if measured else "",
        "log_loss": f"{m['log loss']:.4f}",
        "base_log_loss": f"{m['base log loss']:.4f}",
        "slope": f"{m['slope']:.2f}",
        "raw_slope": f"{raw_slope:.2f}",
        "slices": "\n".join(lines),
        "mid_gap": f"{seg.gap['Mid-market'] * 100:.1f}",
        "smallest": SMALLEST, "rare": RARE,
        "w_may": f"{w[0]:+.3f}", "w_october": f"{w[-1]:+.3f}",
        "largest_missed": round(size.miss["largest"]
                                * size.leavers["largest"]),
        "largest_leavers": int(size.leavers["largest"]),
        "mid_missed": f"{seg.miss['Mid-market']:.0%}",
        "small_missed": f"{seg.miss['Small business']:.0%}"}


def _fit(model, rows: pd.DataFrame):
    """A fresh copy of `model`, fitted on `rows`."""
    return clone(model).fit(rows, rows.not_renewed)
