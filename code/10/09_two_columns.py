# One tree and one forest on two columns, mapped over every value.
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = load(*TRAIN), load(*VALIDATION)
two = ["days_since_order", "log_spend_365"]
X, Xv = features(train)[two].to_numpy(), features(valid)[two].to_numpy()
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()

models = {
    "one tree": DecisionTreeClassifier(random_state=SEED),
    "forest": RandomForestClassifier(n_estimators=300, n_jobs=1,
                                     random_state=SEED)}
gap = np.linspace(0, 200, 81)
spend = np.linspace(6.5, 12.5, 61)
grid = np.array([(a, b) for b in spend for a in gap])
maps = {}
print(f"{'on two columns':<16}{'leaves':>8}{'train AUC':>11}"
      f"{'valid AUC':>11}")
for name, model in models.items():
    model.fit(X, y)
    leaves = (model.get_n_leaves() if name == "one tree" else
              sum(t.get_n_leaves() for t in model.estimators_))
    print(f"{name:<16}{leaves:>8,}"
          f"{auc(y, model.predict_proba(X)[:, 1]):>11.3f}"
          f"{auc(yv, model.predict_proba(Xv)[:, 1]):>11.3f}")
    maps[name] = model.predict_proba(grid)[:, 1].reshape(
        len(spend), len(gap)).round(3).tolist()

with open("code/10/09_two_columns.json", "w") as f:
    json.dump({"gap": gap.tolist(), "spend": spend.tolist(),
               "maps": maps, "leavers": Xv[yv == 1].tolist()}, f)
