# The three splits, and what choosing the best of many lists does.
import json

import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, hits_at_k
from foresight.models.logistic import TRAIN

table = pd.read_parquet(TABLE)
spans = {"training": TRAIN, **SPLITS}
print(f"{'split':11}{'contracts ending':>25}{'rows':>7}{'leavers':>9}"
      f"{'cohorts':>9}")
sizes = {}
for name, (first, last) in spans.items():
    rows = table[table.end_date.between(first, last)]
    # The test year's labels are not read until a milestone's report.
    left = "-" if name == "test" else int(rows.not_renewed.sum())
    sizes[name] = [len(rows), left, rows.moment.nunique()]
    print(f"{name:11}{first:>13} to {last}{len(rows):>7,}"
          f"{left:>9}{sizes[name][2]:>9}")

# 1,000 lists drawn at random, each scored on the validation cohorts.
valid = table[table.end_date.between(*SPLITS["validation"])]
cohorts = [(c.not_renewed.to_numpy(), c.contract_id.to_numpy())
           for _, c in valid.groupby("moment")]
g = rng()
hits = np.array([sum(hits_at_k(y, g.random(len(y)), ids)
                     for y, ids in cohorts) for _ in range(1000)])
calls = 40 * len(cohorts)
print(f"\nA random list, expected: {valid.not_renewed.mean():.1%}"
      f" of {calls} calls")
best_of = {}
for n in (1, 10, 100, 1000):
    best = best_of[n] = int(hits[:n].max())
    print(f"  best of {n:>5,} random lists: {best:>3} leavers,"
          f" {best / calls:.1%}")

with open("code/08/02_three_way_split.json", "w") as f:
    json.dump({"sizes": sizes, "spans": spans, "best_of": best_of,
               "expected": float(valid.not_renewed.mean())}, f)
