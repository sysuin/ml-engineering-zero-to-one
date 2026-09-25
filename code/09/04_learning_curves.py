# Learning curves: the same two models, fitted on more and more rows.
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from foresight.config import TRUTH, rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, known_by
from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, features

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
scale = Standardiser().fit(features(train).to_numpy())
X = scale.transform(features(train).to_numpy())
V = scale.transform(features(valid).to_numpy())
y, yv = train.not_renewed.to_numpy(), valid.not_renewed.to_numpy()


def fit_and_score(A, B, ix):
    """Fit on rows ix of A; AUC on those rows and on all of B."""
    z = Standardiser().fit(A[ix])
    m = LogisticRegression(C=np.inf, tol=1e-8, max_iter=10_000)
    with warnings.catch_warnings():     # tiny samples: no finite best
        warnings.simplefilter("ignore")
        m.fit(z.transform(A[ix]), y[ix])
    return (auc(y[ix], m.decision_function(z.transform(A[ix]))),
            auc(yv, m.decision_function(z.transform(B))))


SIZES = {250: 8, 500: 8, 1000: 5, 1500: 5, 2000: 3, 3000: 3,
         len(train): 1}                 # rows: how many draws
g = rng()
curves = {}
print(f"{'rows':>6}{'16 columns: train':>19}{'valid':>8}"
      f"{'152 columns: train':>20}{'valid':>8}")
for n, draws in SIZES.items():
    picks = [g.choice(len(y), n, replace=False) for _ in range(draws)]
    line = []
    for degree in (1, 2):
        poly = PolynomialFeatures(degree, include_bias=False)
        A, B = poly.fit_transform(X), poly.transform(V)
        s = np.mean([fit_and_score(A, B, ix) for ix in picks], axis=0)
        curves.setdefault(degree, []).append([n, *s])
        line += list(s)
    print(f"{n:>6,}{line[0]:>19.3f}{line[1]:>8.3f}"
          f"{line[2]:>20.3f}{line[3]:>8.3f}")

# The generator's own probabilities: the best any model could rank.
# Read here, for this comparison only; nothing is fitted to them.
p = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
ceiling = {name: auc(rows.not_renewed,
                     p.p_leave[rows.contract_id].to_numpy())
           for name, rows in (("train", train), ("valid", valid))}
print(f"ceiling: training rows {ceiling['train']:.3f},"
      f" validation {ceiling['valid']:.3f}")

with open("code/09/04_learning_curves.json", "w") as f:
    json.dump({"curves": curves, "ceiling": ceiling}, f)
