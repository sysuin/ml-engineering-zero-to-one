# Out-of-bag scores on the training split, used to choose a leaf size.
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train), train.not_renewed.to_numpy()
Xv, yv = features(valid), valid.not_renewed.to_numpy()

print(f"{'smallest leaf':>13}{'OOB AUC':>9}{'OOB top-40':>12}"
      f"{'valid AUC':>11}{'valid top-40':>14}")
for leaf in (1, 5, 10, 25, 50, 100, 200):
    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=leaf, oob_score=True,
        random_state=SEED, n_jobs=1).fit(X, y)
    oob = forest.oob_decision_function_[:, 1]  # trees that missed it
    p = forest.predict_proba(Xv)[:, 1]
    print(f"{leaf:>13}{auc(y, oob):>9.3f}"
          f"{at_capacity(train, oob)['leavers']:>12}"
          f"{auc(yv, p):>11.3f}{at_capacity(valid, p)['leavers']:>14}")
print(f"Training: {len(y):,} contracts in {train.moment.nunique()}"
      f" cohorts; validation: {len(yv):,} in {valid.moment.nunique()}")

print("\nWith leaves of 50, more trees:")
for trees in (50, 100, 300, 1000):
    forest = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=50, oob_score=True,
        random_state=SEED, n_jobs=1).fit(X, y)
    oob = forest.oob_decision_function_[:, 1]
    print(f"  {trees:>5} trees, OOB AUC {auc(y, oob):.4f}")
