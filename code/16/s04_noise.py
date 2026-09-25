# Exercise 4: a column of random numbers given to the booster. Where
# does LightGBM's own importance rank it, and what does shuffling it on
# the validation cohorts cost?
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS
from foresight.explain import by_cohort, permutation_importance, scorer
from foresight.models.featured import FeaturedBooster

table = pd.read_parquet(TABLE)
table["noise"] = rng().random(len(table))
val = SPLITS["validation"]
models = by_cohort(table, *val, lambda: FeaturedBooster(["noise"]))
split = sum(pd.Series(m.model_.booster_.feature_importance("split"),
                      index=m.model_.booster_.feature_name())
            for m in models.values())
share = (split / split.sum()).sort_values(ascending=False)
print("Share of the six boosters' splits")
for rank, (c, v) in enumerate(share.items(), 1):
    print(f"  {rank:>2}  {c:<20}{v:>6.1%}")
rows = table[table.end_date.between(*val)]
imp = permutation_importance(rows, scorer(models),
                             ["noise", "days_since_order"])
for c, r in imp.iterrows():
    print(f"Shuffling {c}: AUC drop {r.auc:.3f}"
          f" ({r.auc_lo:.3f} to {r.auc_hi:.3f})")
