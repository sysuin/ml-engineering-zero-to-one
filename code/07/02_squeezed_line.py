# The sigmoid, and a line squeezed through it to give probabilities.
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from foresight.models.logistic import TRAIN, features, load, sigmoid

print("The sigmoid squeezes any number into (0, 1)")
for z in (-6, -4, -2, -1, 0, 1, 2, 4, 6):
    print(f"  z = {z:>2d}   sigmoid(z) = {sigmoid(z):.4f}")

train = load(*TRAIN)
gap = features(train)[["days_since_order"]].to_numpy()
y = train.not_renewed.to_numpy()
fit = LogisticRegression(C=np.inf).fit(gap, y)   # §7.5 shows how
w, b = fit.coef_[0, 0], fit.intercept_[0]
line = json.load(open("code/07/01_line_fails.json"))

print(f"\nz = {b:.4f} + {w:.5f} x days, probability = sigmoid(z)")
print(f"  {'days':>6}{'z':>9}{'squeezed':>10}{'the line':>10}")
for days in (1, 13, 30, 60, 90, 180, 411, 639):
    z = b + w * days
    straight = line["b"] + line["w"] * days
    print(f"  {days:>6}{z:>+9.3f}{sigmoid(z):>10.1%}"
          f"{straight:>10.1%}")
print(f"Every 30 days without an order multiplies the odds"
      f" by {np.exp(30 * w):.2f}")

with open("code/07/02_squeezed_line.json", "w") as f:
    json.dump({"w": w, "b": b}, f)
