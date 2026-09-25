# The scratch tree with entropy as well as Gini, against scikit-learn.
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.models.logistic import TRAIN, features, load


def gini(pos, n):
    p = pos / n
    return 2 * p * (1 - p)


def entropy(pos, n):
    """Bits, with 0 log 0 taken as 0."""
    p = np.clip(pos / n, 1e-300, 1)
    q = np.clip(1 - pos / n, 1e-300, 1)
    return -(pos / n) * np.log2(p) - (1 - pos / n) * np.log2(q)


def best_split(X, y, impurity):
    n, pos = len(y), y.sum()
    best = (0.0, None, None)
    for col in range(X.shape[1]):
        order = np.argsort(X[:, col], kind="stable")
        x, t = X[order, col], y[order]
        n_left = np.arange(1, n)
        pos_left = np.cumsum(t)[:-1]
        child = (impurity(pos_left, n_left) * n_left
                 + impurity(pos - pos_left, n - n_left)
                 * (n - n_left)) / n
        ok = x[:-1] < x[1:]
        if ok.any():
            drop = np.where(ok, impurity(pos, n) - child, -np.inf)
            i = int(np.argmax(drop))
            if drop[i] > best[0]:
                best = (drop[i], col, (x[i] + x[i + 1]) / 2)
    return best


def grow(X, y, impurity, depth=0, max_depth=4):
    node = {"rate": y.mean()}
    if depth == max_depth or len(y) < 2:
        return node
    drop, col, cut = best_split(X, y, impurity)
    if col is None:
        return node
    left = X[:, col] <= cut
    return node | {"col": col, "cut": cut,
                   "left": grow(X[left], y[left], impurity, depth + 1,
                                max_depth),
                   "right": grow(X[~left], y[~left], impurity,
                                 depth + 1, max_depth)}


def questions(node):
    if "col" not in node:
        return [(-2, -2.0)]
    return ([(node["col"], node["cut"])] + questions(node["left"])
            + questions(node["right"]))


train = load(*TRAIN)
names = list(features(train).columns)
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()


def asks(node):
    if "col" not in node:
        return "(a leaf)"
    return f"{names[node['col']]} <= {node['cut']:g}"


for how, impurity in (("gini", gini), ("entropy", entropy)):
    tree = grow(X, y, impurity)
    ours = questions(tree)
    sk = DecisionTreeClassifier(criterion=how, max_depth=4,
                                random_state=SEED).fit(X, y).tree_
    theirs = list(zip(sk.feature, sk.threshold))
    differ = [i for i, ((a, b), (c, d)) in enumerate(zip(ours, theirs))
              if a != c or not np.isclose(b, d, rtol=1e-5)]
    print(f"{how}: nodes {len(ours)} and {len(theirs)},"
          f" different {len(differ)} {differ}")
    print(f"  first question   {asks(tree)}")
    print(f"  then, if yes     {asks(tree['left'])}")
    print(f"  then, if no      {asks(tree['right'])}")
