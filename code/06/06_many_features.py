# Nine features at once: descent on the raw columns, then standardised.
import json

import numpy as np

from foresight.data.spend import FEATURES, TRAIN, features, spend_table
from foresight.models.linear import Standardiser, descend, squared_error

train = spend_table(*TRAIN)
X, y = features(train), train.spend_next_90d.to_numpy()
print(f"{'column':<18}{'mean':>12}{'std':>12}{'max':>12}")
for name, col in zip(FEATURES, X.T):
    print(f"{name:<18}{col.mean():>12,.1f}{col.std():>12,.1f}"
          f"{col.max():>12,.0f}")


def speed_limit(X):
    """The largest learning rate at which descent does not diverge."""
    A = np.column_stack([X, np.ones(len(X))])
    return 2 / np.linalg.eigvalsh(2 * A.T @ A / len(X)).max()


scaler = Standardiser().fit(X)
Z = scaler.transform(X)
runs = {"raw columns": (X, 0.9 * speed_limit(X), 1.0),
        "standardised": (Z, 0.9 * speed_limit(Z), scaler.scale_)}
print(f"\n{'after 5,000 steps':<16}{'lr':>9}{'RMSE':>8}{'b':>9}"
      f"{'w spend_90d':>13}{'w tickets':>11}")
fitted = {}
for name, (A, lr, scale) in runs.items():
    w, b = descend(A, y, lr, 5000)
    rmse = np.sqrt(squared_error(A, y, w, b))
    if name == "standardised":             # b and w in original units
        b = b - (w / scale) @ scaler.mean_
    w = fitted[name] = w / scale           # per dollar, per ticket, ...
    print(f"{name:<16}{lr:>9.1e}{rmse:>8,.0f}{b:>9,.0f}"
          f"{w[0]:>13.4f}{w[5]:>11.2f}")
moved = abs(fitted["raw columns"]) > 0.01 * abs(fitted["standardised"])
print(f"raw-column weights more than 1% of the way to the answer:"
      f" {moved.sum()} of {len(moved)}")

# For the figure: two columns only, centred so no intercept is needed.
two = X[:, [0, 2]] - X[:, [0, 2]].mean(axis=0)
yc = y - y.mean()
paths = {}
for name, A in (("raw", two), ("standardised", two / two.std(axis=0))):
    lr = 0.5 * 2 / np.linalg.eigvalsh(2 * A.T @ A / len(A)).max()
    w, _, visited = descend(A, yc, lr, 200, path=True)
    paths[name] = {"lr": lr, "path": [list(v) for v, _ in visited],
                   "xx": (A.T @ A / len(A)).tolist(),
                   "xy": (A.T @ yc / len(A)).tolist(),
                   "best": np.linalg.solve(A.T @ A, A.T @ yc).tolist()}
with open("code/06/06_many_features.json", "w") as f:
    json.dump(paths, f)
