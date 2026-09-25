# Logistic regression from scratch: Chapter 6's descend(), new gradient.
import json

import numpy as np

from foresight.models.linear import Standardiser, descend
from foresight.models.logistic import (TRAIN, features, load, log_loss,
                                       log_loss_gradient, sigmoid)

train = load(*TRAIN)
X, y = features(train), train.not_renewed.to_numpy()
print(f"{X.shape[0]:,} rows, {X.shape[1]} columns:")
for i in range(0, X.shape[1], 3):
    names = X.columns[i:i + 3]
    print("  " + "  ".join(f"{c:<20}" for c in names).rstrip())

scaler = Standardiser().fit(X.to_numpy())
Z = scaler.transform(X.to_numpy())
steps, lr = 20_000, 1.0
w, b, path = descend(Z, y, lr, steps, gradient=log_loss_gradient,
                     path=True)

print(f"\nDescent, learning rate {lr}, log loss on the training rows")
path.append((w, b))
for step in (0, 1, 10, 100, 1_000, 5_000, steps):
    ws, bs = path[step]
    loss = log_loss(y, sigmoid(Z @ ws + bs))
    print(f"  step {step:>6,}   log loss {loss:.6f}")
flat = log_loss(y, np.full(len(y), y.mean()))
print(f"Everyone at the training rate: log loss {flat:.6f}")
p = sigmoid(Z @ w + b)
print(f"Mean predicted chance {p.mean():.4f};"
      f" share that left {y.mean():.4f}")

with open("code/07/04_logistic_numpy.json", "w") as f:
    json.dump({"columns": list(X.columns), "w": w.tolist(), "b": b}, f)
