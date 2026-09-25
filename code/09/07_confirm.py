# The chosen strength, read once on the validation backtest.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.logistic import RenewalRisk
from foresight.models.regularised import choose, compare, maker, tune

table = pd.read_parquet(TABLE)
best = choose(tune(table))              # training cohorts only
lasso = backtest(table, *SPLITS["validation"], maker(**best))
v03 = backtest(table, *SPLITS["validation"], RenewalRisk)
rule = v03.assign(model=v03.rule)       # the rule as a "model"

print(f"validation: {lasso.moment.nunique()} cohorts,"
      f" {len(lasso):,} contracts, {lasso.not_renewed.sum()} leavers")
print(f"{'':<22}{'top-40 leavers':>15}{'precision':>11}{'AUC':>8}"
      f"{'log loss':>10}")
for name, s in ((f"lasso {best['strength']:g}", lasso),
                ("v0.3, unpenalised", v03), ("rule", rule)):
    m = measure(s)
    hits = round(m["precision"]["model"] * 240)
    loss = (f"{m['log loss']['model']:>10.4f}" if name != "rule"
            else f"{'-':>10}")
    print(f"{name:<22}{hits:>15}{m['precision']['model']:>11.1%}"
          f"{m['auc']['model']:>8.4f}{loss}")

print("\nlasso minus the other, paired, 2,000 resamples")
print(f"  {'':<8}{'precision, points':<24}AUC")
out = {}
for name, other in (("v0.3", v03), ("rule", rule)):
    c = out[name] = compare(lasso, other)
    (p, plo, phi), (a, alo, ahi) = c["precision"], c["auc"]
    print(f"  {name:<8}{p * 100:+.1f} ({plo * 100:+.1f} to"
          f" {phi * 100:+.1f}){'':<5}{a:+.3f}"
          f" ({alo:+.3f} to {ahi:+.3f})")
with open("code/09/07_confirm.json", "w") as f:
    json.dump(out, f)
