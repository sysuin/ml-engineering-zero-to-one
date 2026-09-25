# Gradient descent for a line, in NumPy: one feature, then the answer.
import json

import numpy as np

from foresight.data.spend import TRAIN, spend_table

train = spend_table(*TRAIN)
x = train.spend_90d.to_numpy() / 1000          # $ thousands, before
y = train.spend_next_90d.to_numpy() / 1000     # $ thousands, after


def loss(w, b):
    """Mean squared error of the line w * x + b."""
    return np.mean((w * x + b - y) ** 2)


def gradient(w, b):
    """Which way is uphill, and how steeply, for w and for b."""
    error = w * x + b - y
    return 2 * np.mean(error * x), 2 * np.mean(error)


w, b = 0.0, 0.0                     # start anywhere: flat, at zero
lr = 0.0015                         # the learning rate: how big a step
path = []
for step in range(5001):
    path.append((w, b, loss(w, b)))
    if step in (0, 1, 2, 3, 10, 100, 1000, 2000, 5000):
        print(f"step {step:>5}   w {w:8.4f}   b {b:8.4f}"
              f"   loss {loss(w, b):9.3f}")
    dw, db = gradient(w, b)
    w, b = w - lr * dw, b - lr * db             # one step downhill

print(f"\nnext quarter = {w:.4f} x last quarter + {b * 1000:,.2f}")
print(f"RMSE on training rows: {np.sqrt(loss(w, b)) * 1000:,.0f}")

# For the figure: the path, and enough to redraw the loss surface.
sums = {"xx": x * x, "x": x, "xy": x * y, "y": y, "yy": y * y}
moments = {k: float(v.mean()) for k, v in sums.items()}
keep = [p for i, p in enumerate(path) if i < 20 or i % 25 == 0]
with open("code/06/04_gradient_descent.json", "w") as f:
    json.dump({"moments": moments, "path": keep, "lr": lr}, f)
