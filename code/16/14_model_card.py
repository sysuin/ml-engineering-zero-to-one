# v0.6's model card: every number measured here, on the validation
# cohorts, and written into docs/foresight-model-card.md.
import json
from pathlib import Path

import pandas as pd

from foresight.card import CARD, write
from foresight.data.build_table import TABLE
from foresight.decide import (COSTS, by_capacity, calibrated,
                              calibration_slope)
from foresight.evaluate import (HISTORY_FROM, REPS, SPLITS, backtest,
                                evaluate, interval, known_by)
from foresight.explain import RARE, SMALLEST, by_cohort
from foresight.slices import key_accounts, label, table
from foresight.score import model
from foresight.train import STRENGTH, make_model

# Measured once from the generator's true chances (DECISIONS.md).
CEILING = {"hits": 64, "precision": "26.7%", "auc": "0.819"}

rows = pd.read_parquet(TABLE)
first, last = SPLITS["validation"]
ev = evaluate("validation", rows, make_model=calibrated(make_model()))
s, pt, dr = ev["scored"], ev["point"], ev["draws"]
raw = backtest(rows, first, last, make_model())
calls = int(s.groupby("moment").size().clip(upper=40).sum())


def cell(what, name, pct):
    v, (lo, hi) = pt[what][name], interval(dr[(what, name)])
    if pct:
        return f"{v:.1%} ({lo:.1%}-{hi:.1%})"
    return f"{v:.3f} ({lo:.3f}-{hi:.3f})"


def pts(v):
    """A share as signed points, with no sign on a zero."""
    v = f"{v * 100:+.1f}"
    return "0.0" if v in ("+0.0", "-0.0") else v


s = label(s, rows[rows.end_date.between("2023-01-01", "2024-06-30")],
          key_accounts())
called = by_capacity(s)
lines = ["| Slice | Contracts | Leavers | Missed (95%) | Renewers"
         " called | Chance minus share left, points (95%) |",
         "|---|---|---|---|---|---|"]
t = {}
for by in ("segment", "size", "population"):
    t[by] = table(s, by, called)
    for name, r in t[by].iterrows():
        miss = ("" if r.leavers == 0 else
                f"{r.miss:.0%} ({r.miss_lo:.0%} to {r.miss_hi:.0%})")
        lines.append(f"| {by}: {name} | {r.contracts:,.0f} |"
                     f" {r.leavers:.0f} | {miss} | {r.fpr:.0%} |"
                     f" {pts(r.gap)} ({pts(r.gap_lo)} to"
                     f" {pts(r.gap_hi)}) |")

keys = key_accounts()
hist = rows[rows.end_date.between(HISTORY_FROM, last)]
kh = hist[hist.account_id.isin(keys)]
h_known = known_by(rows[rows.end_date >= HISTORY_FROM], "2024-04-01")
halloway = model().fit(h_known).predict_proba(
    rows[rows.contract_id == 8118])[0]
lassos = by_cohort(rows, first, last, make_model())
w = [m.weights()["orders_prev_90d"] for m in lassos.values()]
seg, size = t["segment"], t["size"]

numbers = {
    "version": "0.6", "calls": 40, "break_even":
    f"{COSTS.break_even():.1%}", "key_contracts": len(kh),
    "key_left": int(kh.not_renewed.sum()),
    "halloway": f"{halloway:.2g}", "strength": STRENGTH,
    "cohorts": s.moment.nunique(), "first": first, "last": last,
    "contracts": f"{len(s):,}", "leavers": int(s.not_renewed.sum()),
    "calls_total": calls, "reps": f"{REPS:,}",
    "hits": round(pt["precision"]["model"] * calls),
    "rule_hits": round(pt["precision"]["rule"] * calls),
    "ceiling_hits": CEILING["hits"],
    "precision": cell("precision", "model", True),
    "rule_precision": cell("precision", "rule", True),
    "ceiling_precision": CEILING["precision"],
    "auc": cell("auc", "model", False),
    "rule_auc": cell("auc", "rule", False),
    "ceiling_auc": CEILING["auc"],
    "log_loss": f"{pt['log loss']['model']:.4f}",
    "base_log_loss": f"{pt['log loss']['base rate']:.4f}",
    "slope": f"{calibration_slope(s.model, s.not_renewed)[0]:.2f}",
    "raw_slope": f"{calibration_slope(raw.model, raw.not_renewed)[0]:.2f}",
    "slices": "\n".join(lines),
    "mid_gap": f"{seg.gap['Mid-market'] * 100:.1f}",
    "smallest": SMALLEST, "rare": RARE,
    "w_may": f"{w[0]:+.3f}", "w_october": f"{w[-1]:+.3f}",
    "largest_missed": round(size.miss["largest"] * size.leavers[
        "largest"]), "largest_leavers": int(size.leavers["largest"]),
    "mid_missed": f"{seg.miss['Mid-market']:.0%}",
    "small_missed": f"{seg.miss['Small business']:.0%}"}
# The test year's section, once the milestone's read has written it.
read = Path("code/16/15_test_report.json")
text = write(numbers, json.loads(read.read_text()) if read.exists()
             else None)
with open("code/16/14_model_card.json", "w") as f:
    json.dump(numbers, f)

n = numbers
print(f"Validation: {n['cohorts']} cohorts, {n['contracts']} contracts,"
      f" {n['leavers']} leavers, {calls} calls")
print(f"{'':12}{'v0.6':<24}{'rule':<24}{'ceiling':>8}")
print(f"{'leavers':<12}{n['hits']:<24}{n['rule_hits']:<24}"
      f"{n['ceiling_hits']:>8}")
for name, what, pct in (("precision", "precision", True),
                        ("AUC", "auc", False)):
    ceiling = n["ceiling_" + what]
    print(f"{name:<12}{cell(what, 'model', pct):<24}"
          f"{cell(what, 'rule', pct):<24}{ceiling:>8}")
print(f"log loss {n['log_loss']} (base rate {n['base_log_loss']});"
      f" slope {n['slope']} (lasso alone {n['raw_slope']})")
print(f"Key-account contracts ending 2023-24: {n['key_contracts']},"
      f" left {n['key_left']}")
print(f"v0.6's chance for Halloway at its mark: {n['halloway']}")
print(f"Weight on orders_prev_90d: {n['w_may']} in May,"
      f" {n['w_october']} in October")
print(f"Missed: largest quarter {n['largest_missed']} of"
      f" {n['largest_leavers']}; Mid-market {n['mid_missed']},"
      f" small business {n['small_missed']}")
print(f"Written to {CARD.relative_to(CARD.parents[1])}")
