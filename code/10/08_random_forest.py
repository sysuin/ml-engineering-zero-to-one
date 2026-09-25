# Bagging against a random forest: the same trees, fewer columns each.
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
Xv, yv = features(valid).to_numpy(), valid.not_renewed.to_numpy()

kinds = {"bagging": dict(max_features=None),
         "forest": dict(max_features="sqrt"),
         "forest, leaf 50": dict(max_features="sqrt",
                                 min_samples_leaf=50)}
print(f"{'':16}{'columns':>8}{'gap first':>10}{'tree r':>7}"
      f"{'one tree':>9}{'AUC':>7}{'top-40':>7}")
for name, kind in kinds.items():
    model = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                   n_jobs=1, **kind).fit(X, y)
    each = np.array([t.predict_proba(Xv)[:, 1]
                     for t in model.estimators_])
    r = np.corrcoef(each)[np.triu_indices(len(each), 1)].mean()
    one = np.mean([auc(yv, e) for e in each])
    gap = np.mean([t.tree_.feature[0] == 0 for t in model.estimators_])
    p = model.predict_proba(Xv)[:, 1]
    print(f"{name:<16}{model.estimators_[0].max_features_:>8}"
          f"{gap:>10.0%}{r:>7.2f}{one:>9.3f}{auc(yv, p):>7.3f}"
          f"{at_capacity(valid, p)['leavers']:>7}")
print("columns: offered at each split; gap first: trees whose first"
      "\nquestion is days since the last order; tree r: the average"
      "\ncorrelation of two trees' validation predictions; one tree:"
      "\nthe average tree's validation AUC")
