# Accounts on both sides of a split, and folds dealt by account.
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from foresight.config import SEED, rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc
from foresight.models.logistic import RenewalRisk

table = pd.read_parquet(TABLE)
pool = table[table.end_date <= SPLITS["validation"][1]]
pool = pool.reset_index(drop=True)

fit, out = train_test_split(pool, test_size=0.2, random_state=SEED)
seen = out.account_id.isin(fit.account_id)
print(f"Random 80/20 split: {seen.mean():.1%} of held-out contracts"
      "\n  share an account with a training contract")
# What do those accounts' training rows say about the held-out label?
left = fit.groupby("account_id").not_renewed.max()
for name, label in (("leavers", 1), ("renewers", 0)):
    mates = out[seen & (out.not_renewed == label)].account_id
    print(f"  held-out {name:<9}{len(mates):>5,}; account left in a"
          f" training row: {left[mates].mean():.1%}")


def group_folds(accounts, k, seed):
    """Deal accounts, not rows, into k folds: every contract of an
    account lands in the same fold."""
    ids = np.sort(accounts.unique())
    dealt = rng(seed).permutation(ids)
    fold_of = dict(zip(dealt, np.arange(len(ids)) % k))
    return accounts.map(fold_of).to_numpy()


def grouped_scores(seed):
    fold = group_folds(pool.account_id, 5, seed)
    spread = pool.assign(fold=fold).groupby("account_id").fold.nunique()
    assert spread.max() == 1              # no account in two folds
    p = np.zeros(len(pool))
    for f in range(5):
        p[fold == f] = RenewalRisk().fit(pool[fold != f]).predict_proba(
            pool[fold == f])
    return p


later = (pool.moment >= "2023-07-01").to_numpy()
seeds = [auc(pool.not_renewed[later], grouped_scores(SEED + s)[later])
         for s in range(6)]
grouped = seeds[0]
print(f"\nThe same {later.sum():,} contracts, in grouped folds:"
      f" AUC {grouped:.3f}")
print(f"  with six seeds: {min(seeds):.3f} to {max(seeds):.3f}")

with open("code/08/07_grouped_split.json", "w") as f:
    json.dump({"grouped": grouped, "seeds": seeds,
               "overlap": float(seen.mean())}, f)
