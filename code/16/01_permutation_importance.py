# Permutation importance on the validation cohorts: each column shuffled
# within each cohort, 50 times, for v0.5's lasso and the booster; then
# the booster's own importance, measured on the rows it learned from.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS
from foresight.explain import (COLUMNS, by_cohort,
                               permutation_importance, scorer)
from foresight.models.boosting import RenewalBooster
from foresight.train import make_model

table = pd.read_parquet(TABLE)
rows = table[table.end_date.between(*SPLITS["validation"])]
val = SPLITS["validation"]
fitted = {"lasso": by_cohort(table, *val, make_model()),
          "booster": by_cohort(table, *val, RenewalBooster)}
imp = {name: permutation_importance(rows, scorer(models), COLUMNS)
       for name, models in fitted.items()}

# LightGBM's importance: splits made, and log loss removed, per column,
# summed over the six cohorts' boosters and shown as shares.
split, gain = 0, 0
for m in fitted["booster"].values():
    b = m.model_.booster_
    split = split + pd.Series(b.feature_importance("split"),
                              index=b.feature_name())
    gain = gain + pd.Series(b.feature_importance("gain"),
                            index=b.feature_name())
split, gain = split / split.sum(), gain / gain.sum()



def num(v):
    """Three places, and no minus sign on a zero."""
    return f"{v:6.3f}".replace("-0.000", " 0.000")


order = imp["lasso"].auc.sort_values(ascending=False).index
for name, d in imp.items():
    extra = f"{'splits':>8}{'gain':>6}" if name == "booster" else ""
    print(f"{name:<19}{'AUC drop':>8}{'95% range':>17}{'lost':>6}"
          + extra)
    for c in order:
        r = d.loc[c]
        cells = (f"{split[c]:>8.0%}{gain[c]:>6.0%}"
                 if name == "booster" else "")
        print(f"  {c:<17}  {num(r.auc)}  {num(r.auc_lo)} to"
              f"{num(r.auc_hi)}{r.leavers:>6.1f}{cells}")
    print()
print("AUC drop and leavers lost from 240 calls: mean over 50")
print("resamples, each shuffled afresh. splits, gain: LightGBM's own")
print("shares, from the rows each booster was fitted on.")

with open("code/16/01_permutation_importance.json", "w") as f:
    json.dump({"order": list(order),
               "imp": {n: d.round(5).reset_index().to_dict("records")
                       for n, d in imp.items()},
               "split": split.round(4).to_dict(),
               "gain": gain.round(4).to_dict()}, f)
