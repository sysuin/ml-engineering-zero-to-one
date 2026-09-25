# The forward pass, one step at a time, on one real contract.
import json

import numpy as np

from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
cols = ["orders_90d", "orders_prev_90d"]
X = features(train)[cols]
row = train.index[train.contract_id == 7332][0]
raw = X.loc[row]
x = ((raw - X.mean()) / X.std(ddof=0)).round(2).to_numpy()
print(f"Contract 7332: orders in the last 90 days {raw.iloc[0]:.0f},"
      f" in the 90 before {raw.iloc[1]:.0f}")
print(f"  standardised: x1 = {x[0]:+.2f}, x2 = {x[1]:+.2f}")

# Weights chosen by hand: small, round, easy to follow.
W1 = np.array([[-1.0, 0.5],        # from x1 to units 1 and 2
               [1.0, 0.5]])        # from x2 to units 1 and 2
b1 = np.array([0.2, -0.1])
v, c = np.array([2.0, -1.0]), -2.5

z1 = x @ W1 + b1
h = np.maximum(z1, 0)
for j in range(2):
    print(f"unit {j + 1}: {W1[0, j]:+.1f} x {x[0]:+.2f}"
          f" {W1[1, j]:+.1f} x {x[1]:+.2f} {b1[j]:+.1f}"
          f" = {z1[j]:+.2f}   ReLU -> {h[j]:.2f}")
z = h @ v + c
p = 1 / (1 + np.exp(-z))
print(f"output: {v[0]:+.1f} x {h[0]:.2f} {v[1]:+.1f} x {h[1]:.2f}"
      f" {c:+.1f} = {z:+.2f}")
print(f"sigmoid({z:+.2f}) = {p:.4f}, the chance of not renewing")
y = int(train.not_renewed[row])
print(f"It did not renew (y = {y}): log loss -log({p:.4f})"
      f" = {-np.log(p):.4f}")

with open("code/19/04_forward_by_hand.json", "w") as f:
    json.dump({"x": x.tolist(), "W1": W1.tolist(), "b1": b1.tolist(),
               "v": v.tolist(), "c": c, "y": y}, f)
