# Two models scored on the rows they learned from and on later rows.
from sklearn.neighbors import KNeighborsClassifier

from foresight.evaluate import auc
from foresight.models.linear import Standardiser
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       at_capacity, days_since_rule,
                                       features, load)

train, valid = load(*TRAIN), load(*VALIDATION)
scale = Standardiser().fit(features(train).to_numpy())


def columns(rows):
    return scale.transform(features(rows).to_numpy())


# Nearest match: find the most similar training contract, copy its
# outcome. A lookup, not a model of anything.
nearest = KNeighborsClassifier(n_neighbors=1)
nearest.fit(columns(train), train.not_renewed)
logistic = RenewalRisk().fit(train)
scorers = {
    "nearest match": lambda r: nearest.predict_proba(columns(r))[:, 1],
    "logistic": logistic.predict_proba,
    "rule": days_since_rule,
}
print(f"{'':15}{'rows':>12}{'AUC':>8}{'top-40 precision':>19}")
for name, score in scorers.items():
    for split, rows in (("training", train), ("validation", valid)):
        s = score(rows)
        p = at_capacity(rows, s)["precision"]
        print(f"{name:15}{split:>12}{auc(rows.not_renewed, s):>8.3f}"
              f"{p:>19.1%}")
