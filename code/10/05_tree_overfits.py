# One tree grown until it stops: what it learned, and what it knows.
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
X, Xv = features(train), features(valid)
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()

print(f"{'depth':>5}{'leaves':>8}{'train AUC':>11}{'valid AUC':>11}"
      f"{'top-40 leavers':>16}")
for depth in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    tree.fit(X, y)
    p = tree.predict_proba(Xv)[:, 1]
    print(f"{depth or 'all':>5}{tree.get_n_leaves():>8}"
          f"{auc(y, tree.predict_proba(X)[:, 1]):>11.3f}"
          f"{auc(yv, p):>11.3f}{at_capacity(valid, p)['leavers']:>16}")

# The last tree, grown until every leaf is pure.
leaf = tree.apply(X)
size = np.bincount(leaf)[np.unique(leaf)]
rate = np.bincount(leaf, y)[np.unique(leaf)] / size
pure = np.isin(rate, (0, 1)).sum()
print(f"\nGrown to the end: depth {tree.get_depth()},"
      f" {tree.get_n_leaves()} leaves, {pure} of them pure")
print(f"  leaves holding one training contract: {(size == 1).sum()},"
      f" median leaf {np.median(size):.0f}")
sure = p == 0
print(f"  validation contracts it gives no chance of leaving:"
      f" {sure.sum():,}")
print(f"  of those, left anyway: {yv[sure].sum()} of {yv.sum()}"
      f" leavers")
