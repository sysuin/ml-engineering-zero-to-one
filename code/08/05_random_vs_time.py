# One model, one set of contracts: split at random, then by time.
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, measure
from foresight.models.logistic import RenewalRisk, days_since_rule

table = pd.read_parquet(TABLE)
pool = table[table.end_date <= SPLITS["validation"][1]]
pool = pool.reset_index(drop=True)      # 2023 and 2024: no test year

# Twenty random 80/20 splits: the same model, a different 20% each time.
single = []
for s in range(20):
    fit, out = train_test_split(pool, test_size=0.2,
                                random_state=SEED + s)
    p = RenewalRisk().fit(fit).predict_proba(out)
    single.append(auc(out.not_renewed, p))
q = np.percentile(single, [0, 50, 100])
print(f"20 random splits, AUC on the held-out 20%: lowest {q[0]:.3f},"
      f"\n  median {q[1]:.3f}, highest {q[2]:.3f}")

# Now score the same contracts both ways: every cohort from July 2023.
# The first six months of outcomes are history for the time split.
later = (pool.moment >= "2023-07-01").to_numpy()


def random_folds(seed):
    """Five random folds: every contract scored by a model that did not
    see it, but did see contracts from before and after it."""
    p = np.zeros(len(pool))
    folds = KFold(5, shuffle=True, random_state=seed)
    for fit, out in folds.split(pool):
        p[out] = RenewalRisk().fit(pool.iloc[fit]).predict_proba(
            pool.iloc[out])
    return p


def earlier_cohorts():
    """Each cohort scored by a model fitted on the cohorts before it."""
    p = np.full(len(pool), np.nan)
    for mark in pool.moment[later].unique():
        now = (pool.moment == mark).to_numpy()
        fit = pool[pool.moment < mark]
        p[now] = RenewalRisk().fit(fit).predict_proba(pool[now])
    return p


rows = pool[later]
ways = {"random folds": random_folds(SEED), "time": earlier_cohorts(),
        "the rule": days_since_rule(pool)}
print(f"\nThe same {len(rows):,} contracts, {rows.moment.nunique()} "
      f"cohorts, {rows.not_renewed.sum()} leavers")
print(f"  {'scored by':<14}{'AUC':>7}{'top-40 leavers':>16}"
      f"{'precision':>11}")
same = {}
for name, p in ways.items():
    m = measure(rows.assign(model=p[later], rule=0.0, base=0.1))
    same[name] = [m["auc"]["model"], m["precision"]["model"]]
    hits = round(m["precision"]["model"] * 40 * rows.moment.nunique())
    print(f"  {name:<14}{same[name][0]:>7.3f}{hits:>16}"
          f"{same[name][1]:>11.1%}")
seeds = [auc(rows.not_renewed, random_folds(SEED + s)[later])
         for s in range(1, 6)]
print(f"Random folds with five other seeds: AUC {min(seeds):.3f}"
      f" to {max(seeds):.3f}")

with open("code/08/05_random_vs_time.json", "w") as f:
    json.dump({"single": single, "same": same, "seeds": seeds,
               "rows": len(rows)}, f)
