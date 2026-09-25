# Without an activation, two layers are one layer in disguise.
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from foresight.config import SEED, rng
from foresight.models.linear import Standardiser
from foresight.models.logistic import (TRAIN, VALIDATION, features,
                                       load, log_loss)

train, valid = load(*TRAIN), load(*VALIDATION)
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
scaler = Standardiser().fit(X)
Z = scaler.transform(X)
Zv = scaler.transform(features(valid).to_numpy())
yv = valid.not_renewed.to_numpy()

# Two layers of random weights, 16 -> 8 -> 1, and no activation.
g = rng()
W1, b1 = g.normal(size=(16, 8)), g.normal(size=8)
W2, b2 = g.normal(size=8), g.normal()
deep = (Z @ W1 + b1) @ W2 + b2
w, b = W1 @ W2, b1 @ W2 + b2           # multiply the layers out
flat = Z @ w + b
print(f"Two layers vs one: largest gap {np.abs(deep - flat).max():.1e}")
relu = np.maximum(Z @ W1 + b1, 0) @ W2 + b2
best = np.linalg.lstsq(np.c_[Z, np.ones(len(Z))], relu, rcond=None)[0]
miss = relu - np.c_[Z, np.ones(len(Z))] @ best
print("With ReLU, the nearest single layer misses by"
      f" {np.abs(miss).mean():.3f} on average")

# Now trained: the same shape of network, with and without ReLU.
warnings.simplefilter("ignore", ConvergenceWarning)
print(f"\n{'log loss':<30}{'params':>8}{'training':>10}"
      f"{'validation':>12}")


def show(name, model):
    n = sum(c.size for c in model.coefs_ + model.intercepts_) \
        if hasattr(model, "coefs_") else model.coef_.size + 1
    a = log_loss(y, model.predict_proba(Z)[:, 1])
    b = log_loss(yv, model.predict_proba(Zv)[:, 1])
    print(f"{name:<30}{n:>8}{a:>10.4f}{b:>12.4f}")


show("one neuron (Chapter 7)",
     LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(Z, y))
for act in ("identity", "relu"):
    net = MLPClassifier(hidden_layer_sizes=(8, 8), activation=act,
                        solver="lbfgs", alpha=0, max_iter=5_000,
                        random_state=SEED).fit(Z, y)
    show(f"16 -> 8 -> 8 -> 1, {act}", net)
