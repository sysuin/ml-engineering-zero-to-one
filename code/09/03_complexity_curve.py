# From memorising to averaging everything: the lookup's one dial.
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, known_by
from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, at_capacity, features

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
scale = Standardiser().fit(features(train).to_numpy())
X = scale.transform(features(train).to_numpy())
V = scale.transform(features(valid).to_numpy())
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()

print("k nearest matches vote: the score is the share who left")
print(f"{'k':>6}{'train AUC':>11}{'valid AUC':>11}"
      f"{'valid top-40':>14}")
curve = []
for k in (1, 3, 10, 30, 100, 300, 1000, 3000, len(train)):
    vote = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    s, sv = vote.predict_proba(X)[:, 1], vote.predict_proba(V)[:, 1]
    hits = at_capacity(valid, sv)["leavers"]
    curve.append([k, auc(y, s), auc(yv, sv), hits])
    print(f"{k:>6,}{auc(y, s):>11.3f}{auc(yv, sv):>11.3f}{hits:>14}")

print("\nLogistic regression, unpenalised")
print(f"{'columns':>8}{'':>10}{'train AUC':>11}{'valid AUC':>11}"
      f"{'valid top-40':>14}")
for degree, name in ((1, "Chapter 7"), (2, "+ pairs")):
    poly = PolynomialFeatures(degree, include_bias=False)
    A, B = poly.fit_transform(X), poly.transform(V)
    z = Standardiser().fit(A)
    m = LogisticRegression(C=np.inf, tol=1e-8, max_iter=10_000)
    m.fit(z.transform(A), y)
    s = m.decision_function(z.transform(A))
    sv = m.decision_function(z.transform(B))
    hits = at_capacity(valid, sv)["leavers"]
    print(f"{A.shape[1]:>8}{name:>10}{auc(y, s):>11.3f}"
          f"{auc(yv, sv):>11.3f}{hits:>14}")

with open("code/09/03_complexity_curve.json", "w") as f:
    json.dump(curve, f)
