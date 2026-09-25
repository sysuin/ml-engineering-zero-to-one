# The region columns: signal, or four more places for luck to enter?
import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, backtest, known_by
from foresight.models.logistic import TRAIN, at_capacity, log_loss
from foresight.models.regularised import (TUNE, RegularisedRisk,
                                          choose, compare, maker, tune)

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
left = {n: r.groupby("region").not_renewed.mean()
        for n, r in (("training", train), ("validation", valid))}
print(f"{'share not renewing':<20}{'training':>10}{'validation':>12}")
for region in left["training"].index:
    print(f"  {region:<18}{left['training'][region]:>10.1%}"
          f"{left['validation'][region]:>12.1%}")

# Refit on 300 resamples of accounts: how firmly is each weight known?
g, by = rng(), train.groupby("account_id").indices
ids = list(by)
w = [RegularisedRisk("l2", 0.0).fit(train.iloc[np.concatenate(
     [by[a] for a in g.choice(ids, len(ids))])]).weights()
     for _ in range(300)]
w = pd.DataFrame(w).filter(like="region=")
print("\nregion weights, 95% of resamples (reference: Southwest)")
for c in w:
    lo, hi = np.percentile(w[c], [2.5, 97.5])
    print(f"  {c:<18}{lo:+.3f} to {hi:+.3f}")

best = choose(tune(table))
drop = ("region=",)
print(f"\n{'':<24}{'tuning cohorts':^18}{'validation':^25}".rstrip())
print(f"{'':<24}{'AUC':>8}{'log loss':>10}{'AUC':>8}{'log loss':>10}"
      f"{'top 40':>7}")
runs = {}
lasso = (best["penalty"], best["strength"])
for name, pen, s, d in (("v0.3", "l2", 0.0, ()),
                        ("v0.3, no regions", "l2", 0.0, drop),
                        ("lasso", *lasso, ()),
                        ("lasso, no regions", *lasso, drop)):
    line = f"{name:<24}"
    for first, last in (TUNE, SPLITS["validation"]):
        s_ = runs[name, first] = backtest(table, first, last,
                                          maker(pen, s, d))
        y, p = s_.not_renewed.to_numpy(), s_.model.to_numpy()
        line += f"{auc(y, p):>8.4f}{log_loss(y, p):>10.4f}"
    print(line + f"{at_capacity(s_, s_.model)['leavers']:>7}")

print("\nv0.3 without regions minus v0.3, paired AUC")
for name, first in (("tuning cohorts", TUNE[0]),
                    ("validation", SPLITS["validation"][0])):
    d, lo, hi = compare(runs["v0.3, no regions", first],
                        runs["v0.3", first])["auc"]
    print(f"  {name:<16}{d:+.4f} ({lo:+.4f} to {hi:+.4f})")
