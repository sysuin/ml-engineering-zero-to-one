# Three kinds of noise, ranked by impurity importance.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from foresight.config import SEED, rng
from foresight.models.forest import COLUMNS, MIN_LEAF, TREES
from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
X, y = features(train), train.not_renewed.to_numpy()
g = rng()
X["noise, any number"] = g.random(len(X))
X["noise, 1 to 10"] = g.integers(1, 11, len(X))
X["noise, coin flip"] = g.integers(0, 2, len(X))

forest = RandomForestClassifier(
    n_estimators=TREES, min_samples_leaf=MIN_LEAF, max_features=COLUMNS,
    random_state=SEED, n_jobs=1).fit(X, y)
ranked = pd.Series(forest.feature_importances_, X.columns) \
    .sort_values(ascending=False)
print(f"{'column':<24}{'rank':>5}{'importance':>12}{'values':>8}")
for col in ranked.index:
    if col.startswith("noise") or col == "discount_pct":
        rank = list(ranked.index).index(col) + 1
        print(f"{col:<24}{rank:>5}{ranked[col]:>12.3f}"
              f"{X[col].nunique():>8,}")
print(f"({len(ranked)} columns in all)")
