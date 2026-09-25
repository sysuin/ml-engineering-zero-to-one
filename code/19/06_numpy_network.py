# A two-layer network in NumPy, trained on the renewal table.
import json

import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import (RenewalRisk, at_capacity,
                                       days_since_rule, features,
                                       log_loss, sigmoid)
from foresight.models.mlp import known_split

train, valid = known_split(pd.read_parquet(TABLE))
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
scaler = Standardiser().fit(X)
Z = scaler.transform(X)
Zv = scaler.transform(features(valid).to_numpy())


def forward(X, W1, b1, v, c):
    z1 = X @ W1 + b1                    # 16 columns -> 8 sums
    h = np.maximum(z1, 0)               # ReLU: 8 learned features
    return z1, h, sigmoid(h @ v + c)    # 8 features -> 1 chance


def backward(X, y, W1, v, z1, h, p):
    """Each parameter's gradient: the blame, averaged over rows."""
    dz = (p - y) / len(y)
    dz1 = np.outer(dz, v) * (z1 > 0)
    return X.T @ dz1, dz1.sum(axis=0), h.T @ dz, dz.sum()


g = rng()
W1 = g.normal(0, np.sqrt(2 / 16), (16, 8))   # small and random
b1, v, c = np.zeros(8), g.normal(0, np.sqrt(1 / 8), 8), 0.0
start = {"W1": W1.tolist(), "b1": b1.tolist(), "v": v.tolist(), "c": c}

lr, steps = 0.5, 3_000
print(f"{'step':>6}{'training loss':>15}{'validation loss':>17}")
for step in range(steps + 1):
    z1, h, p = forward(Z, W1, b1, v, c)
    if step in (0, 1, 10, 100, 1_000, 2_000, steps):
        pv = forward(Zv, W1, b1, v, c)[2]
        print(f"{step:>6,}{log_loss(y, p):>15.4f}"
              f"{log_loss(valid.not_renewed, pv):>17.4f}")
    if step < steps:
        dW1, db1, dv, dc = backward(Z, y, W1, v, z1, h, p)
        W1, b1, v, c = W1 - lr * dW1, b1 - lr * db1, v - lr * dv, \
            c - lr * dc

print(f"\n{len(train):,} training rows, {len(valid):,} validation")
print(f"{'top 40 of each cohort':<24}{'leavers':>8}{'precision':>11}")
lists = [("network", forward(Zv, W1, b1, v, c)[2]),
         ("Chapter 7's model",
          RenewalRisk().fit(train).predict_proba(valid)),
         ("the rule", days_since_rule(valid))]
for name, score in lists:
    s = at_capacity(valid, score)
    print(f"{name:<24}{s['leavers']:>8}{s['precision']:>11.1%}")

with open("code/19/06_numpy_network.json", "w") as f:
    json.dump({"start": start, "lr": lr, "steps": steps,
               "end": {"W1": W1.tolist(), "b1": b1.tolist(),
                       "v": v.tolist(), "c": c}}, f)
