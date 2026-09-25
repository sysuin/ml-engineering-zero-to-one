# Bagging from scratch: trees on bootstrap samples, averaged.
import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED, rng
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
Xv, yv = features(valid).to_numpy(), valid.not_renewed.to_numpy()
g = rng()
n = len(y)

bags, votes = [], []
for b in range(300):
    pick = g.integers(0, n, n)          # n rows, drawn with replacement
    tree = DecisionTreeClassifier(random_state=SEED)   # no limits
    tree.fit(X[pick], y[pick])
    bags.append(np.bincount(pick, minlength=n))
    votes.append(tree.predict_proba(Xv)[:, 1])
votes = np.array(votes)

print(f"{'trees':>5}{'valid AUC':>11}{'top-40 leavers':>16}")
for b in (1, 5, 25, 100, 300):
    p = votes[:b].mean(axis=0)          # the bag's average opinion
    print(f"{b:>5}{auc(yv, p):>11.3f}"
          f"{at_capacity(valid, p)['leavers']:>16}")
alone = [auc(yv, v) for v in votes]
print(f"One tree at a time: valid AUC {min(alone):.3f} to"
      f" {max(alone):.3f}, mean {np.mean(alone):.3f}")

bags = np.array(bags)
left_out = (bags == 0).mean(axis=1)
print(f"Rows a bootstrap sample leaves out: {left_out.mean():.1%}"
      f" ({left_out.min():.1%} to {left_out.max():.1%})")

with open("code/10/07_bagging.json", "w") as f:
    json.dump({"counts": bags[:6, :14].tolist(),
               "rows": train.contract_id[:14].tolist(),
               "left_out": float(left_out.mean())}, f)
