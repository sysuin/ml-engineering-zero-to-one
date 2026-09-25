# K-means by hand on two features, then scikit-learn from that start.
import json

import numpy as np
from sklearn.cluster import KMeans

from foresight.config import rng
from foresight.segments import (SEGMENT_DATE, account_features,
                                assign, inertia, load, update)

orders, accounts = load()
f = account_features(SEGMENT_DATE, orders, accounts)
raw = np.column_stack([np.log(f.orders_365), np.log(f.recency_days)])
X = (raw - raw.mean(axis=0)) / raw.std(axis=0)
print(f"{len(X):,} accounts ordered in the year before {SEGMENT_DATE}")

k = 3
centres = X[rng().choice(len(X), size=k, replace=False)]
history = [centres]
print(f"\n{'step':>4}{'inertia':>10}   sizes")
for step in range(1, 101):
    labels = assign(X, centres)             # 1. nearest centre
    new = update(X, labels, k)              # 2. centre = mean
    history.append(new)
    done = (assign(X, new) == labels).all()   # nobody moved
    if step <= 3 or step % 10 == 0 or done:
        sizes = np.bincount(labels, minlength=k)
        print(f"{step:>4}{inertia(X, labels, centres):>10.1f}   "
              f"{sizes}")
    centres = new
    if done:
        break

km = KMeans(k, init=history[0], n_init=1, tol=0).fit(X)
same = (km.labels_ == assign(X, centres)).mean()
gap = np.abs(np.sort(km.cluster_centers_, axis=0)
             - np.sort(centres, axis=0)).max()
print(f"\nscikit-learn from the same start: inertia {km.inertia_:.1f}")
print(f"same segment for {same:.1%} of accounts; "
      f"largest centre gap {gap:.1e}")

print(f"\n{'centre':>6}{'orders':>9}{'days since':>12}{'accounts':>10}")
labels = assign(X, centres)
for j in np.argsort(centres[:, 0]):
    back = np.exp(centres[j] * raw.std(axis=0) + raw.mean(axis=0))
    print(f"{j:>6}{back[0]:>9.1f}{back[1]:>12.1f}"
          f"{(labels == j).sum():>10,}")

show = rng().choice(len(X), size=700, replace=False)
json.dump({"points": raw[show].round(4).tolist(),
           "mean": raw.mean(axis=0).tolist(),
           "std": raw.std(axis=0).tolist(),
           "centres": [c.round(4).tolist() for c in history],
           "labels": [assign(X[show], c).tolist() for c in history]},
          open("code/18/01_kmeans_steps.json", "w"))
