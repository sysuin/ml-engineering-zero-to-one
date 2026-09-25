# Exercise 1: a scratch booster built LightGBM's way, against LightGBM
# itself, ten rounds of depth-3 trees on the training split.
import numpy as np
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.models.logistic import (TRAIN, VALIDATION, features,
                                       load, sigmoid)

train, valid = load(*TRAIN), load(*VALIDATION)
X, V = features(train).to_numpy(), features(valid).to_numpy()
y = train.not_renewed.to_numpy()
ROUNDS, RATE, DEPTH = 10, 0.1, 3


def best_split(A, g, h):
    """The column and cut with the largest gain G_L^2/H_L + G_R^2/H_R
    - G^2/H, LightGBM's split score with no penalty."""
    best = (0.0, None, None)
    G, H = g.sum(), h.sum()
    for j in range(A.shape[1]):
        order = np.argsort(A[:, j], kind="stable")
        a = A[order, j]
        gl, hl = np.cumsum(g[order])[:-1], np.cumsum(h[order])[:-1]
        ok = a[1:] > a[:-1]                 # a real gap between values
        gain = gl ** 2 / hl + (G - gl) ** 2 / (H - hl) - G ** 2 / H
        gain = np.where(ok, gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best[0]:
            best = (gain[i], j, a[i])       # left: value <= a[i]
    return best


def grow(A, g, h, depth):
    if depth == DEPTH:
        return {"value": -g.sum() / h.sum()}
    gain, j, cut = best_split(A, g, h)
    if j is None:
        return {"value": -g.sum() / h.sum()}   # the Newton step
    left = A[:, j] <= cut
    return {"col": j, "cut": cut,
            "left": grow(A[left], g[left], h[left], depth + 1),
            "right": grow(A[~left], g[~left], h[~left], depth + 1)}


def predict_tree(node, A):
    if "value" in node:
        return np.full(len(A), node["value"])
    left = A[:, node["col"]] <= node["cut"]
    out = np.empty(len(A))
    out[left] = predict_tree(node["left"], A[left])
    out[~left] = predict_tree(node["right"], A[~left])
    return out


start = np.log(y.mean() / (1 - y.mean()))
F, trees = np.full(len(y), start), []
for _ in range(ROUNDS):
    p = sigmoid(F)
    tree = grow(X, p - y, p * (1 - p), 0)   # gradient and hessian
    F += RATE * predict_tree(tree, X)
    trees.append(tree)

lgbm = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=ROUNDS,
                      learning_rate=RATE, max_depth=DEPTH, num_leaves=8,
                      min_child_samples=1, min_child_weight=0,
                      reg_lambda=0, max_bin=1023,
                      min_data_in_bin=1).fit(X, y)
mine = {"training": sigmoid(F),
        "validation": sigmoid(start + RATE * sum(predict_tree(t, V)
                                                 for t in trees))}
theirs = {"training": lgbm.predict_proba(X)[:, 1],
          "validation": lgbm.predict_proba(V)[:, 1]}


def questions(node, mine=True):
    """The columns a tree asks about, in order, root first."""
    if ("value" if mine else "leaf_value") in node:
        return []
    kids = ("left", "right") if mine else ("left_child", "right_child")
    col = node["col"] if mine else node["split_feature"]
    return [col] + questions(node[kids[0]], mine) + questions(
        node[kids[1]], mine)


theirs_trees = [t["tree_structure"] for t in
                lgbm.booster_.dump_model()["tree_info"]]
root = theirs_trees[0]
print(f"Root split: mine column {trees[0]['col']} at "
      f"{trees[0]['cut']:g}, LightGBM's column "
      f"{root['split_feature']} at {root['threshold']:g}")
same = [questions(a) == questions(b, False)
        for a, b in zip(trees, theirs_trees)]
print(f"Trees asking the same columns in the same places: "
      f"{sum(same)} of {ROUNDS}; first to differ: round "
      f"{same.index(False) + 1 if False in same else '-'}")
for split in ("training", "validation"):
    gap = np.abs(mine[split] - theirs[split])
    print(f"{split:<11} largest gap {gap.max():.1e},"
          f" rows differing by over 1e-9: {(gap > 1e-9).sum():,}")
