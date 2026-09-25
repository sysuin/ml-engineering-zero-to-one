# Boosting on one column: each tree fits what the last ones left over.
import json

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from foresight.config import SEED
from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
gap = features(train)[["days_since_order"]].to_numpy()
y = train.not_renewed.to_numpy()

RATE = 0.5                  # how much of each tree's answer to keep
SHOWN = (0, 1, 2, 5, 50)    # rounds to print
guess = np.full(len(y), y.mean())       # round 0: everyone the same
kept, trees = {0: guess.copy()}, []
for r in range(1, max(SHOWN) + 1):
    left_over = y - guess                       # the residuals
    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED)
    tree.fit(gap, left_over)
    guess = guess + RATE * tree.predict(gap)
    kept[r] = guess.copy()
    trees.append(tree)

print(f"Round 0 gives every contract the training share,"
      f" {y.mean():.1%}")
first = trees[0].tree_
cuts = sorted(t for t in first.threshold if t > 0)
print("Tree 1 cuts the gap at", ", ".join(f"{c:g}" for c in cuts),
      "days")
bands = [(0, 30), (31, 60), (61, 90), (91, 180), (181, 365)]
print(f"\n{'gap, days':<12}{'contracts':>10}{'left':>7}"
      + "".join(f"{'round ' + str(r):>9}" for r in SHOWN[1:]))
for lo, hi in bands:
    b = (gap[:, 0] >= lo) & (gap[:, 0] <= hi)
    print(f"{lo:>4} to {hi:<4}{b.sum():>10,}{y[b].mean():>7.1%}"
          + "".join(f"{kept[r][b].mean():>9.1%}" for r in SHOWN[1:]))
print(f"\n{'round':>5}{'mean |left over|':>18}{'squared error':>15}")
for r in SHOWN:
    print(f"{r:>5}{np.abs(y - kept[r]).mean():>18.4f}"
          f"{np.mean((y - kept[r]) ** 2):>15.5f}")

xs = np.arange(0, 366)
curve = {r: [float(y.mean())] * len(xs) if r == 0 else
         (y.mean() + RATE * sum(t.predict(xs[:, None].astype(float))
                                for t in trees[:r])).tolist()
         for r in SHOWN}
inside = [(gap[:, 0] >= lo) & (gap[:, 0] <= hi) for lo, hi in bands]
with open("code/11/01_left_over.json", "w") as f:
    json.dump({"x": xs.tolist(), "curves": curve, "rate": RATE,
               "bands": [[lo, hi, float(y[b].mean())]
                         for (lo, hi), b in zip(bands, inside)],
               "band_guess": {r: [float(kept[r][b].mean())
                                  for b in inside] for r in SHOWN}}, f)
