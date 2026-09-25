# Does a long gap mean more for a small business? The interaction in
# the rates, written as columns for the lasso, and left to the trees.
import pandas as pd

from foresight.features.build import load
from foresight.features.registry import names
from foresight.models.featured import (TRAINING, lasso, tuning_score,
                                       watched_loss)

table = load()
train = table[table.end_date.between(*TRAINING)]
gap = pd.cut(train.days_since_order.astype(float).fillna(365),
             [0, 14, 45, 90, 999],
             labels=["up to 14", "15 to 45", "46 to 90", "over 90"])
small = train.segment.eq("Small business").map(
    {True: "small business", False: "the rest"})
r = train.pivot_table(index=gap, columns=small, values="not_renewed",
                      aggfunc="mean", observed=True)
print(f"{'Training, days since order':<28}{'small business':>15}"
      f"{'the rest':>10}{'ratio':>7}")
for band, row in r.iterrows():
    print(f"  {band:<26}{row['small business']:>15.1%}"
          f"{row['the rest']:>10.1%}"
          f"{row['small business'] / row['the rest']:>7.1f}")

pairs = names("interactions")
print(f"\n{'Tuning cohorts, lasso':<30}{'log loss':>9}{'AUC':>7}"
      f"{'leavers':>9}")
for name, extra in (("v0.4's columns", ()),
                    ("  + two interactions", pairs)):
    s = tuning_score(table, lasso(extra))
    print(f"  {name:<28}{s['log loss']:>9.5f}{s['auc']:>7.3f}"
          f"{s['hits']:>9}")

print(f"\n{'Watched months, booster':<30}{'watched':>9}{'rounds':>8}")
for name, extra, leaves in (("stumps", (), 2),
                            ("stumps + two interactions", pairs, 2),
                            ("trees of 4 leaves", (), 4),
                            ("4 leaves + two interactions", pairs, 4)):
    loss, rounds = watched_loss(table, extra, leaves, 50)
    print(f"  {name:<28}{loss:>9.5f}{rounds:>8,}")
