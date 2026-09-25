# Exercise 1: a third learning curve, for a model with one column.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, known_by
from foresight.models.logistic import TRAIN, features

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
X = features(train)[["days_since_order"]].to_numpy()
V = features(valid)[["days_since_order"]].to_numpy()
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()

g = rng()
print(f"{'rows':>6}{'train AUC':>11}{'valid AUC':>11}")
for n, draws in ((250, 8), (500, 8), (1000, 5), (2000, 3),
                 (len(train), 1)):
    s = []
    for _ in range(draws):
        ix = g.choice(len(y), n, replace=False)
        m = LogisticRegression(C=np.inf).fit(X[ix], y[ix])
        s.append((auc(y[ix], m.decision_function(X[ix])),
                  auc(yv, m.decision_function(V))))
    a, b = np.mean(s, axis=0)
    print(f"{n:>6,}{a:>11.3f}{b:>11.3f}")
