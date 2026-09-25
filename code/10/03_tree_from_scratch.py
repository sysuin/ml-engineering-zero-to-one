# A decision tree from scratch, then scikit-learn's on the same rows.
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.models.logistic import TRAIN, VALIDATION, features, load


def gini(pos, n):
    """Gini impurity from a count of leavers and a count of rows."""
    p = pos / n
    return 2 * p * (1 - p)


def best_split(X, y, min_leaf):
    """The column and cut whose split lowers weighted Gini the most."""
    n, pos = len(y), y.sum()
    best = (0.0, None, None)                    # (decrease, col, cut)
    for col in range(X.shape[1]):
        order = np.argsort(X[:, col], kind="stable")
        x, t = X[order, col], y[order]
        n_left = np.arange(1, n)                # rows left of each gap
        pos_left = np.cumsum(t)[:-1]
        child = (gini(pos_left, n_left) * n_left
                 + gini(pos - pos_left, n - n_left) * (n - n_left)) / n
        ok = ((x[:-1] < x[1:]) & (n_left >= min_leaf)
              & (n - n_left >= min_leaf))       # a real gap, big sides
        if ok.any():
            drop = np.where(ok, gini(pos, n) - child, -np.inf)
            i = int(np.argmax(drop))
            if drop[i] > best[0]:
                best = (drop[i], col, (x[i] + x[i + 1]) / 2)
    return best


def grow(X, y, depth=0, max_depth=3, min_leaf=1, min_drop=0.0,
         n_all=None):
    """A node: a leaf with its leaver rate, or a question and two
    smaller trees. Stops at max_depth, when a node is too small to
    split, or when the best split improves too little."""
    n_all = n_all or len(y)
    node = {"rows": len(y), "rate": y.mean()}
    if depth == max_depth or len(y) < 2 * min_leaf:
        return node
    drop, col, cut = best_split(X, y, min_leaf)
    if col is None or drop * len(y) / n_all < min_drop:
        return node
    left = X[:, col] <= cut
    rules = dict(max_depth=max_depth, min_leaf=min_leaf,
                 min_drop=min_drop, n_all=n_all)
    return node | {
        "col": col, "cut": cut,
        "left": grow(X[left], y[left], depth + 1, **rules),
        "right": grow(X[~left], y[~left], depth + 1, **rules)}


def predict(node, x):
    """Follow the questions from the root to a leaf."""
    while "col" in node:
        node = node["left"] if x[node["col"]] <= node["cut"] else \
            node["right"]
    return node["rate"]


def questions(node):
    """Every (column, cut) in the order sklearn numbers its nodes."""
    if "col" not in node:
        return [(-2, -2.0)]
    return ([(node["col"], node["cut"])] + questions(node["left"])
            + questions(node["right"]))


train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
Xv = features(valid).to_numpy()
for depth, leaf, least in ((3, 1, 0), (6, 20, 0), (8, 20, 1e-4),
                          (8, 5, 0)):
    mine = grow(X, y, max_depth=depth, min_leaf=leaf, min_drop=least)
    sk = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                min_impurity_decrease=least,
                                random_state=SEED).fit(X, y)
    ours = questions(mine)
    theirs = list(zip(sk.tree_.feature, sk.tree_.threshold))
    same = sum(a == c and np.isclose(b, d, rtol=1e-5)
               for (a, b), (c, d) in zip(ours, theirs))
    p = np.array([predict(mine, x) for x in Xv])
    q = sk.predict_proba(Xv)[:, 1]
    print(f"Depth {depth}, leaf {leaf}, least drop {least:g}: nodes"
          f" {len(ours)} and {len(theirs)}, identical {same}")
    print(f"  validation predictions equal: {np.isclose(p, q).sum():,}"
          f" of {len(p):,}; largest gap {np.abs(p - q).max():.1e}")

# The last pair parts company. Where, and why?
i = next(i for i, (a, b) in enumerate(zip(ours, theirs))
         if a[0] != b[0] or not np.isclose(a[1], b[1], rtol=1e-5))
at = sk.decision_path(X)[:, i].toarray().ravel() == 1
Xi, yi = X[at], y[at]
names = list(features(train).columns)
print(f"First difference: node {i}, {len(yi)} rows,"
      f" {yi.sum()} leavers")
for who, (col, cut) in (("ours", ours[i]), ("sklearn", theirs[i])):
    L = Xi[:, col] <= cut
    after = (gini(yi[L].sum(), L.sum()) * L.sum()
             + gini(yi[~L].sum(), (~L).sum()) * (~L).sum()) / len(yi)
    print(f"  {who:<8}{names[col]} <= {cut:g}: Gini falls"
          f" {gini(yi.sum(), len(yi)) - after:.5f}")
