# Partial dependence on days since the last order: every validation
# contract given the same gap, scored by the lasso and the booster
# fitted on the outcomes known at the last validation mark.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, SPLITS, known_by
from foresight.explain import partial_dependence
from foresight.models.boosting import RenewalBooster
from foresight.train import make_model

table = pd.read_parquet(TABLE)
rows = table[table.end_date.between(*SPLITS["validation"])]
mark = rows.moment.max()
known = known_by(table[table.end_date >= HISTORY_FROM], mark)
lasso = make_model()().fit(known)
booster = RenewalBooster().fit(known)

grid = [1, 7, 14, 30, 45, 60, 90, 120, 150, 180, 240, 300, 365]
pd_lasso = partial_dependence(lasso, rows, "days_since_order", grid)
pd_boost = partial_dependence(booster, rows, "days_since_order", grid)

# What the rows the models learned from say, gap by gap.
bands = [0, 14, 30, 60, 90, 180, 420]
gap = pd.cut(known.days_since_order.astype(float), bands)
seen = known.groupby(gap, observed=True).not_renewed.agg(["size",
                                                           "mean"])

print("Average chance with every validation contract's gap set to")
print(f"{'days':>6}{'lasso':>9}{'booster':>9}")
for d in grid:
    print(f"{d:>6}{pd_lasso[d]:>9.1%}{pd_boost[d]:>9.1%}")
print(f"\nThe rows learned from{'rows':>12}{'left':>7}")
for band, r in seen.iterrows():
    print(f"  gap {int(band.left):>3} to {int(band.right):<3} days"
          f"{int(r['size']):>12,}{r['mean']:>7.1%}")
recent = (rows.orders_90d > 0).mean()
print(f"\nValidation contracts with an order in the last 90 days:"
      f" {recent:.1%}")
print(f"Largest gap among the {len(known):,} rows learned from:"
      f" {known.days_since_order.max()} days")

with open("code/16/02_partial_dependence.json", "w") as f:
    json.dump({"grid": grid, "lasso": list(pd_lasso.round(5)),
               "booster": list(pd_boost.round(5)),
               "bands": [[int(b.left), int(b.right)]
                         for b in seen.index],
               "rows": [int(n) for n in seen["size"]],
               "left": [round(float(v), 5) for v in seen["mean"]]}, f)
