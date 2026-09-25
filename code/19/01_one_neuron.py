# One neuron, trained the way networks are, is Chapter 7's model.
import json

import numpy as np

from foresight.config import rng
from foresight.models.linear import Standardiser
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       features, load, sigmoid,
                                       top_of_each_cohort)

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train), train.not_renewed.to_numpy()
scaler = Standardiser().fit(X.to_numpy())
Z = scaler.transform(X.to_numpy())


def neuron(x, w, b):
    """Weigh each input, add them up with a bias, squash."""
    return sigmoid(x @ w + b)


# A network starts from small random weights, not from zero.
w, b = rng().normal(0, 0.1, Z.shape[1]), 0.0
for step in range(20_000):
    error = neuron(Z, w, b) - y          # Chapter 7's p - y
    w -= 1.0 * Z.T @ error / len(y)
    b -= 1.0 * error.mean()

ch7 = RenewalRisk().fit(train)           # Chapter 7's model, as built
lib = ch7.model_
print(f"{'':22}{'neuron':>10}{'Chapter 7':>11}")
print(f"{'bias / intercept':<22}{b:>+10.4f}{lib.intercept_[0]:>+11.4f}")
for name, a, c in list(zip(X.columns, w, lib.coef_[0]))[:5]:
    print(f"{name:<22}{a:>+10.4f}{c:>+11.4f}")
print(f"... and {len(w) - 5} more weights")
gap = max(np.abs(w - lib.coef_[0]).max(),
          abs(b - lib.intercept_[0]))
print(f"Largest difference, all 17 numbers: {gap:.1e}")

Zv = scaler.transform(features(valid).to_numpy())
p_neuron, p_ch7 = neuron(Zv, w, b), ch7.predict_proba(valid)
print(f"\nValidation, {len(valid):,} contracts")
print(f"  largest gap in probability: "
      f"{np.abs(p_neuron - p_ch7).max():.1e}")
mine = set(top_of_each_cohort(valid, p_neuron).contract_id)
theirs = set(top_of_each_cohort(valid, p_ch7).contract_id)
print(f"  top-40 lists: {len(mine & theirs)} of {len(mine)} calls"
      " the same")

with open("code/19/01_one_neuron.json", "w") as f:
    json.dump({"columns": list(X.columns), "w": w.tolist(), "b": b}, f)
