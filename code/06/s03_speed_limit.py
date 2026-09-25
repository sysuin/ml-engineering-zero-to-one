# Exercise 3: find by trial the learning rate where descent diverges.
import numpy as np

from foresight.data.spend import TRAIN, features, spend_table
from foresight.models.linear import Standardiser, descend

train = spend_table(*TRAIN)
y = train.spend_next_90d.to_numpy() / 1000
problems = {
    "one feature ($k)": train[["spend_90d"]].to_numpy() / 1000,
    "nine, standardised": Standardiser().fit(
        features(train)).transform(features(train)),
}


def diverges(X, lr, steps=3000):
    with np.errstate(over="ignore", invalid="ignore"):
        w, b = descend(X, y, lr, steps)
    return not np.isfinite(b) or abs(b) > 1e6


for name, X in problems.items():
    lo, hi = 0.0, 1.0                      # lo converges, hi diverges
    for _ in range(30):                    # halve the gap thirty times
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if diverges(X, mid) else (mid, hi)
    A = np.column_stack([X, np.ones(len(X))])
    theory = 2 / np.linalg.eigvalsh(2 * A.T @ A / len(X)).max()
    print(f"{name:<20} by trial {lo:.6f}   from the curvature"
          f" {theory:.6f}")
