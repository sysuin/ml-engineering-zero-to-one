# Two correlated columns: the lasso's weights on the two order counts
# in each validation cohort's fit, the counts shuffled alone and
# together, and the lasso refitted without each.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, backtest,
                                known_by)
from foresight.explain import (by_cohort, hits, permutation_importance,
                               scorer)
from foresight.models.featured import gain
from foresight.models.regularised import maker
from foresight.train import make_model

table = pd.read_parquet(TABLE)
val = SPLITS["validation"]
rows = table[table.end_date.between(*val)]
models = by_cohort(table, *val, make_model())
pair = ["orders_90d", "orders_prev_90d"]

print("Weights per standard deviation, each cohort's fit")
print(f"{'mark':<12}{'rows':>7}{'orders_90d':>12}{'prev_90d':>10}"
      f"{'sum':>8}{'corr':>7}")
weights = []
for mark, m in models.items():
    w = m.weights()
    known = known_by(table[table.end_date >= HISTORY_FROM], mark)
    corr = known[pair].astype(float).corr().iloc[0, 1]
    weights.append([str(mark.date()), float(w[pair[0]]),
                    float(w[pair[1]])])
    print(f"{mark:%Y-%m-%d}{len(known):>9,}{w[pair[0]]:>12.3f}"
          f"{w[pair[1]]:>10.3f}{w[pair].sum():>8.3f}{corr:>7.2f}")

imp = permutation_importance(rows, scorer(models),
                             [pair[0], pair[1], tuple(pair)])
print(f"\nShuffled{'AUC drop':>30}{'95% range':>17}{'lost':>6}")
for c, r in imp.iterrows():
    print(f"  {c:<26}{r.auc:>10.3f}  {r.auc_lo:>6.3f} to"
          f"{r.auc_hi:>6.3f}{r.leavers:>6.1f}")

print(f"\nRefitted without{'leavers':>22}{'AUC':>8}")
refits = {}
for drop in [(), (pair[0],), (pair[1],), tuple(pair),
             ("days_since_order",)]:
    s = backtest(table, *val, maker("l1", 0.002, drop))
    refits[drop] = s
    name = " and ".join(drop) if drop else "(nothing: v0.5)"
    print(f"  {name:<36}{hits(s, s.model):>4}"
          f"{auc(s.not_renewed, s.model):>8.3f}")
d, lo, hi = gain(refits[tuple(pair)], refits[()])["auc"]
print(f"Without both, minus v0.5: AUC {d:+.3f}"
      f" ({lo:+.3f} to {hi:+.3f})")

with open("code/16/03_correlated_credit.json", "w") as f:
    json.dump({"weights": weights}, f)
