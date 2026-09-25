# Impurity importance, with a column of pure noise added to the table.
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED, rng
from foresight.models.forest import COLUMNS, MIN_LEAF, TREES
from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
X, y = features(train), train.not_renewed.to_numpy()
X["noise"] = rng().random(len(X))     # a random number, nothing more

ranked = {}
for name, leaf in (("Foresight's forest", MIN_LEAF),
                   ("leaves of 1", 1)):
    forest = RandomForestClassifier(
        n_estimators=TREES, min_samples_leaf=leaf, max_features=COLUMNS,
        random_state=SEED, n_jobs=1).fit(X, y)
    ranked[name] = pd.Series(forest.feature_importances_, X.columns) \
        .sort_values(ascending=False)

a, b = ranked
print(f"{'':4}{a + ', leaf ' + str(MIN_LEAF):<32}{b}")
for i in range(9):
    cells = [f"{s.index[i]:<22}{s.iloc[i]:>6.3f}"
             for s in ranked.values()]
    print(f"{i + 1:>2}  {cells[0]:<32}{cells[1]}")
for name, s in ranked.items():
    where = list(s.index).index("noise") + 1
    print(f"{name}: noise ranks {where} of {len(s)},"
          f" above {len(s) - where} real columns")

with open("code/10/13_importance_trap.json", "w") as f:
    json.dump({k: s.round(4).to_dict() for k, s in ranked.items()}, f)
