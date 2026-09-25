# Rescaled columns change nothing; an interaction costs nothing.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from foresight.config import SEED
from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = load(*TRAIN), load(*VALIDATION)
X, Xv = features(train).to_numpy(), features(valid).to_numpy()
y = train.not_renewed.to_numpy()
scale = Standardiser().fit(X)
views = {"as they are": lambda A: A,
         "standardised": scale.transform,
         "cubed, plus 1,000": lambda A: A ** 3 + 1000}
base = None
for name, view in views.items():
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                  random_state=SEED).fit(view(X), y)
    p = tree.predict_proba(view(Xv))[:, 1]
    base = p if base is None else base
    cut = tree.tree_.threshold[0]
    print(f"{name:<18} first cut {cut:>12,.4f}"
          f"   same predictions {np.isclose(p, base).mean():.0%}")

# Two questions: a long gap, and a small discount. Four cells.
gap = (features(train).days_since_order > 92).astype(int)
low = (features(train).discount_pct <= 4).astype(int)
cells = pd.DataFrame({"gap > 92": gap, "discount <= 4": low, "y": y})
two = cells[["gap > 92", "discount <= 4"]].to_numpy()
additive = LogisticRegression(C=np.inf).fit(two, y)
tree = DecisionTreeClassifier(max_depth=2, random_state=SEED)
tree.fit(two, y)
cells["line"] = additive.predict_proba(two)[:, 1]
cells["tree"] = tree.predict_proba(two)[:, 1]
words = {0: "no", 1: "yes"}
cells[["gap > 92", "discount <= 4"]] = \
    cells[["gap > 92", "discount <= 4"]].replace(words)
table = cells.groupby(["gap > 92", "discount <= 4"]).agg(
    contracts=("y", "size"), left=("y", "mean"),
    logistic=("line", "mean"), tree=("tree", "mean"))
print("\nShare leaving, and each model's estimate, by cell")
print(table.to_string(formatters={c: "{:.1%}".format for c in
                                  ("left", "logistic", "tree")}))
