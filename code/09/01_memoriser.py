# The nearest-match lookup again: why it is perfect on its own rows.
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, known_by
from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, at_capacity, features

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
first_mark = valid.moment.min()
# Only outcomes on record at the first validation mark: no gap.
train = known_by(table[table.end_date.between(*TRAIN)], first_mark)
print(f"learns from {len(train):,} contracts known by"
      f" {first_mark:%d %B %Y}; scored on {len(valid):,}")

scale = Standardiser().fit(features(train).to_numpy())
X = scale.transform(features(train).to_numpy())
V = scale.transform(features(valid).to_numpy())
lookup = NearestNeighbors(n_neighbors=1).fit(X)
y = train.not_renewed.to_numpy()

d, i = lookup.kneighbors(X)             # each training row's match
own = (i[:, 0] == np.arange(len(X))).mean()
print(f"training rows whose nearest match is themselves: {own:.1%}")

dv, iv = lookup.kneighbors(V)           # each validation row's match
copy = y[iv[:, 0]]                      # the outcome it copies
same = (train.account_id.to_numpy()[iv[:, 0]]
        == valid.account_id.to_numpy())
print(f"validation rows matched to their own account:"
      f" {same.mean():.1%}")
print(f"median distance to the match: training {np.median(d):.2f},"
      f" validation {np.median(dv):.2f}")

yv = valid.not_renewed.to_numpy()
print(f"\n{'':12}{'accuracy':>10}{'AUC':>8}{'top-40 leavers':>16}")
for name, rows, truth, guess in (("training", train, y, y[i[:, 0]]),
                                 ("validation", valid, yv, copy)):
    hits = at_capacity(rows, guess)["leavers"]
    print(f"{name:12}{(guess == truth).mean():>10.1%}"
          f"{auc(truth, guess):>8.3f}{hits:>16}")
print(f"{'call nobody':12}{1 - yv.mean():>10.1%}")
