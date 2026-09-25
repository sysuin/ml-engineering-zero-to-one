# A depth-3 tree read aloud, then refitted on resampled rows.
import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED, rng
from foresight.evaluate import auc
from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train), train.not_renewed.to_numpy()


def between(lo, hi, unit="", fewer="fewer"):
    """A count's range as words; the cuts fall between whole numbers."""
    if lo == -np.inf:
        return f"{int(hi)}{unit} or {fewer}"
    if hi == np.inf:
        return f"{int(lo) + 1}{unit} or more"
    return f"{int(lo) + 1} to {int(hi)}{unit}"


def phrase(col, lo, hi):
    """What a leaf's contracts have in common, for one column."""
    if "=" in col:                          # a 0/1 category column
        return ("" if lo >= 0.5 else "not ") + col.split("=")[1]
    if col == "days_since_order":
        return f"last order {between(lo, hi)} days ago"
    if col == "discount_pct":
        return f"discount {between(lo, hi, '%', 'less')}"
    if col == "orders_90d":
        return f"{between(lo, hi)} orders in 90 days"
    if col == "orders_prev_90d":
        return f"{between(lo, hi)} orders in the 90 days before"
    return f"{col} in ({lo:g}, {hi:g}]"


def leaves(t, names, node=0, box=None):
    """Every leaf as (leaver rate, rows, what its rows share)."""
    box = box or {}
    if t.children_left[node] < 0:
        said = [phrase(c, *box[c]) for c in box]
        rows = int(t.n_node_samples[node])
        return [(t.value[node][0][1], rows, said)]
    col, cut = names[t.feature[node]], t.threshold[node]
    lo, hi = box.get(col, (-np.inf, np.inf))
    return (leaves(t, names, t.children_left[node],
                   box | {col: (lo, min(hi, cut))})
            + leaves(t, names, t.children_right[node],
                     box | {col: (max(lo, cut), hi)}))


tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X, y)
print(f"{'left':>6}{'of':>7}   the contracts in the leaf")
for rate, rows, said in sorted(leaves(tree.tree_, list(X.columns)),
                               key=lambda r: -r[0]):
    print(f"{rate:6.1%}{rows:>7,}   {said[0]}")
    for more in said[1:]:
        print(f"{'':16}{more}")

# The same tree, fitted on five resamples of the same training rows.
SHORT = {"days_since_order": "gap", "orders_90d": "orders",
         "orders_prev_90d": "orders before", "discount_pct": "discount",
         "log_spend_365": "log spend", "tenure_days": "tenure",
         "legacy_terms": "legacy"}


def asks(t, node):
    """A node's question, in short form."""
    col = X.columns[t.feature[node]]
    return f"{SHORT.get(col, col)} <= {t.threshold[node]:.3g}"


def top3(t):
    return [asks(t, 0), asks(t, t.children_left[0]),
            asks(t, t.children_right[0])]


g = rng()
print(f"\n{'fitted on':<11}{'first question':<15}{'then, if yes':<19}"
      f"{'if no':<18}{'AUC':>5}")
fits = [("training", np.arange(len(y)))]
fits += [(f"resample {b}", g.integers(0, len(y), len(y)))
         for b in range(1, 6)]
for name, pick in fits:
    other = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    other.fit(X.iloc[pick], y[pick])
    score = auc(valid.not_renewed,
                other.predict_proba(features(valid))[:, 1])
    a, b, c = top3(other.tree_)
    print(f"{name:<11}{a:<15}{b:<19}{c:<18}{score:.3f}")

t = tree.tree_
nodes = [{"col": X.columns[t.feature[i]] if t.feature[i] >= 0 else None,
          "cut": float(t.threshold[i]),
          "rows": int(t.n_node_samples[i]),
          "rate": float(t.value[i][0][1]),
          "left": int(t.children_left[i]),
          "right": int(t.children_right[i])}
         for i in range(t.node_count)]
with open("code/10/06_read_a_tree.json", "w") as f:
    json.dump(nodes, f)
