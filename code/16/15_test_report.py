# The milestone's one look at the test year: v0.6, the lasso with its
# chances calibrated, against the rule; its chances beside v0.5's; its
# errors by slice; and the result added to the model card.
# timeout: 300
import json

import pandas as pd

from foresight.card import write
from foresight.config import ROOT
from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, calibrated, calibration_slope
from foresight.evaluate import (SPLITS, auc, backtest, evaluate,
                                report)
from foresight.models.logistic import log_loss
from foresight.models.regularised import compare
from foresight.slices import key_accounts, label, page
from foresight.train import make_model

rows = pd.read_parquet(TABLE)
PAGE = ROOT / "docs" / "foresight-v0.6-test.md"
ev = evaluate("test", rows, make_model=calibrated(make_model()))
print(report(ev, PAGE,
             "Foresight v0.6 on the test year: reported once"))

s = ev["scored"]
raw = backtest(rows, *SPLITS["test"], make_model())
y = s.not_renewed.to_numpy()
keys = key_accounts()
called = by_capacity(s)
in_list = int(s[called].account_id.isin(keys).sum())
lines = ["", f"{'':22}{'log loss':>10}{'slope':>8}{'average':>10}",
         f"  {'share that left':<20}{'':>18}{y.mean():>10.1%}"]
for name, p in (("v0.5, uncalibrated", raw.model),
                ("v0.6, calibrated", s.model)):
    lines.append(f"  {name:<20}{log_loss(y, p.to_numpy()):>10.4f}"
                 f"{calibration_slope(p, y)[0]:>8.2f}{p.mean():>10.1%}")
d, lo, hi = compare(s, raw)["auc"]
within = {n: sum(auc(c.not_renewed, c[n]) for _, c in
                 s.assign(raw=raw.model).groupby("moment")) / 12
          for n in ("raw", "model")}
lines += [f"  AUC, v0.6 minus v0.5, paired: {d:+.3f}"
          f" ({lo:+.3f} to {hi:+.3f})",
          f"  AUC within each cohort, averaged: v0.5"
          f" {within['raw']:.3f}, v0.6 {within['model']:.3f}",
          f"  key accounts in the top 40s: {in_list}", "",
          "The test year is now spent for v0.6."]
print("\n".join(lines))

train = rows[rows.end_date.between("2023-01-01", "2024-06-30")]
sliced = page(label(s, train, keys), called,
              slices=["segment", "size", "population"])
print("\n" + sliced)
with open(PAGE, "a") as f:
    f.write("\n```text" + "\n".join(lines) + "\n\n" + sliced
            + "\n```\n")

with open("code/16/14_model_card.json") as f:
    numbers = json.load(f)
pt = ev["point"]
calls = int(s.groupby("moment").size().clip(upper=40).sum())
dp, lop, hip = (v * 100 for v in (
    pt["precision"]["model"] - pt["precision"]["rule"],
    *pd.Series(ev["draws"][("precision", "model")]
               - ev["draws"][("precision", "rule")]).quantile(
        [0.025, 0.975])))
test = {"version": "0.6", "first": SPLITS["test"][0],
        "last": SPLITS["test"][1], "contracts": f"{len(s):,}",
        "leavers": int(y.sum()), "calls_total": calls, "calls": 40,
        "hits": round(pt["precision"]["model"] * calls),
        "rule_hits": round(pt["precision"]["rule"] * calls),
        "ceiling_hits": 173, "auc": f"{pt['auc']['model']:.3f}",
        "rule_auc": f"{pt['auc']['rule']:.3f}",
        "log_loss": f"{log_loss(y, s.model.to_numpy()):.4f}",
        "raw_log_loss": f"{log_loss(y, raw.model.to_numpy()):.4f}",
        "slope": f"{calibration_slope(s.model, y)[0]:.2f}",
        "raw_slope": f"{calibration_slope(raw.model, y)[0]:.2f}",
        "diff": f"{dp:+.1f} points ({lop:+.1f} to {hip:+.1f})"}
write(numbers, test)
with open("code/16/15_test_report.json", "w") as f:
    json.dump(test, f)
