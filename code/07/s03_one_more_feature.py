# Exercise 3: add Chapter 6's spend_90d, and score it against the rule.
import numpy as np
from sklearn.linear_model import LogisticRegression

from foresight.data.spend import spend_table
from foresight.models.linear import Standardiser
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       days_since_rule, features)

train, valid = spend_table(*TRAIN), spend_table(*VALIDATION)


def grid(rows, extra):
    X = features(rows)
    if extra:
        X["log_spend_90d"] = np.log1p(rows.spend_90d)
    return X.to_numpy()


print(f"  {'':22}{'leavers':>8}{'precision':>11}{'recall':>8}")
for name, extra in (("sixteen columns", False),
                    ("plus log spend_90d", True)):
    s = Standardiser().fit(grid(train, extra))
    model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000)
    model.fit(s.transform(grid(train, extra)), train.not_renewed)
    p = model.predict_proba(s.transform(grid(valid, extra)))[:, 1]
    r = at_capacity(valid, p)
    print(f"  {name:22}{r['leavers']:>8}{r['precision']:>11.1%}"
          f"{r['recall']:>8.1%}")
r = at_capacity(valid, days_since_rule(valid))
print(f"  {'days-since rule':22}{r['leavers']:>8}"
      f"{r['precision']:>11.1%}{r['recall']:>8.1%}")
