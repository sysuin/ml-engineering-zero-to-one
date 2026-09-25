# A gradient booster in thirty lines, then checked against the library.
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load, log_loss,
                                       sigmoid)

train, valid = load(*TRAIN), load(*VALIDATION)
X, V = features(train).to_numpy(), features(valid).to_numpy()
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()
ROUNDS, RATE, DEPTH = 100, 0.1, 3


def shallow_tree():
    return DecisionTreeRegressor(max_depth=DEPTH, random_state=SEED)


# ------------------------------------ version 1: fit the residuals
def boost_residuals(X, y):
    """Each tree fits y minus the probability so far."""
    start, trees = y.mean(), []
    guess = np.full(len(y), start)
    for _ in range(ROUNDS):
        tree = shallow_tree().fit(X, y - guess)
        guess = guess + RATE * tree.predict(X)
        trees.append(tree)
    return start, trees


def predict_residuals(model, A):
    start, trees = model
    return start + RATE * sum(t.predict(A) for t in trees)


# ------------------------------------ version 2: fit the gradient
def boost_log_loss(X, y):
    """Work in log-odds. Each tree fits the negative gradient of log
    loss, and each leaf takes a Newton step."""
    start, trees = np.log(y.mean() / (1 - y.mean())), []
    F = np.full(len(y), start)                  # log-odds so far
    for _ in range(ROUNDS):
        p = sigmoid(F)
        tree = shallow_tree().fit(X, y - p)     # the negative gradient
        leaf = tree.apply(X)
        step = {j: (y - p)[leaf == j].sum()
                / (p * (1 - p))[leaf == j].sum() for j in set(leaf)}
        F = F + RATE * np.array([step[j] for j in leaf])
        trees.append((tree, step))
    return start, trees


def predict_log_loss(model, A):
    start, trees = model
    return sigmoid(start + RATE * sum(
        np.array([step[j] for j in tree.apply(A)])
        for tree, step in trees))


# ------------------------------------ against scikit-learn
v1, v2 = boost_residuals(X, y), boost_log_loss(X, y)
library = GradientBoostingClassifier(
    n_estimators=ROUNDS, learning_rate=RATE, max_depth=DEPTH,
    random_state=SEED).fit(X, y)

gap = abs(predict_log_loss(v2, X) - library.predict_proba(X)[:, 1])
print(f"Training rows: largest gap from the library {gap.max():.1e}")
same = sum(np.array_equal(a.tree_.feature, b.tree_.feature)
           and np.allclose(a.tree_.threshold, b.tree_.threshold)
           for (a, _), b in zip(v2[1], library.estimators_[:, 0]))
print(f"Trees with the same questions and cuts: {same} of {ROUNDS}")
gap = abs(predict_log_loss(v2, V) - library.predict_proba(V)[:, 1])
print(f"Validation contracts scored differently: "
      f"{(gap > 1e-9).sum()} of {len(yv):,}, by up to {gap.max():.3f}")

print(f"\n{'validation':<22}{'AUC':>7}{'log loss':>10}"
      f"{'top-40 leavers':>16}{'below 0':>9}")
for name, p in (("version 1, residuals", predict_residuals(v1, V)),
                ("version 2, log loss", predict_log_loss(v2, V)),
                ("scikit-learn", library.predict_proba(V)[:, 1])):
    ll = f"{log_loss(yv, p):>10.4f}" if p.min() > 0 else f"{'-':>10}"
    print(f"{name:<22}{auc(yv, p):>7.4f}{ll}"
          f"{at_capacity(valid, p)['leavers']:>16}{(p < 0).sum():>9}")
