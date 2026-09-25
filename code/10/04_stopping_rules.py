# Three ways to stop a tree growing, and what each does to it.
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
X, Xv = features(train), features(valid)
settings = [("none", {})]
settings += [(f"depth {d}", {"max_depth": d}) for d in (2, 4, 6, 8)]
settings += [(f"leaf {n}", {"min_samples_leaf": n})
             for n in (5, 25, 100, 200)]
settings += [(f"drop {g:g}", {"min_impurity_decrease": g})
             for g in (1e-4, 3e-4, 1e-3)]

print(f"{'stopping rule':<14}{'leaves':>7}{'train AUC':>11}"
      f"{'valid AUC':>11}{'top-40 leavers':>16}")
for name, limit in settings:
    tree = DecisionTreeClassifier(random_state=SEED, **limit)
    tree.fit(X, train.not_renewed)
    p = tree.predict_proba(Xv)[:, 1]
    fit = auc(train.not_renewed, tree.predict_proba(X)[:, 1])
    print(f"{name:<14}{tree.get_n_leaves():>7}{fit:>11.3f}"
          f"{auc(valid.not_renewed, p):>11.3f}"
          f"{at_capacity(valid, p)['leavers']:>16}")

print("\nGini or entropy, with the same limits")
for limit in ({"max_depth": 4}, {"min_samples_leaf": 100}):
    line = []
    for how in ("gini", "entropy"):
        tree = DecisionTreeClassifier(criterion=how, random_state=SEED,
                                      **limit).fit(X, train.not_renewed)
        p = tree.predict_proba(Xv)[:, 1]
        line.append(f"{how} {auc(valid.not_renewed, p):.3f}")
    rule = ", ".join(f"{k} {v}" for k, v in limit.items())
    print(f"  {rule:<22}valid AUC: {line[0]}, {line[1]}")
