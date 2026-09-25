# Log loss: what one prediction costs, and what a whole list costs.
import json

import numpy as np

from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       features, load, log_loss,
                                       sigmoid)

print("One contract that did not renew (y = 1)")
print(f"  {'predicted':>10}{'log loss':>10}{'squared error':>15}")
for p in (0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001):
    print(f"  {p:>10}{-np.log(p):>10.3f}{(1 - p) ** 2:>15.3f}")

train, valid = load(*TRAIN), load(*VALIDATION)
y = valid.not_renewed.to_numpy()
fit = json.load(open("code/07/02_squeezed_line.json"))
z = fit["b"] + fit["w"] * features(valid).days_since_order.to_numpy()
lists = {
    "training rate, everyone":
        np.full(len(y), train.not_renewed.mean()),
    "days since order": sigmoid(z),
    "same order, sure of it": sigmoid(3 * z + 4),
}
print(f"\nValidation, {len(y):,} contracts")
print(f"  {'':25}{'log loss':>9}{'top-40 precision':>18}")
for name, p in lists.items():
    top = at_capacity(valid, p)["precision"]
    shown = "  (no order)" if np.ptp(p) == 0 else f"{top:>18.1%}"
    print(f"  {name:25}{log_loss(y, p):>9.4f}{shown}")
print(f"  {'':25}{'on leavers':>11}{'on renewers':>13}")
for name in ("days since order", "same order, sure of it"):
    p = lists[name]
    print(f"  {name:25}{log_loss(y[y == 1], p[y == 1]):>11.3f}"
          f"{log_loss(y[y == 0], p[y == 0]):>13.3f}")

# For the figure: the loss curve for each outcome.
grid = np.linspace(0.001, 0.999, 999)
with open("code/07/03_log_loss.json", "w") as f:
    json.dump({"p": grid.tolist(), "yes": (-np.log(grid)).tolist(),
               "no": (-np.log(1 - grid)).tolist()}, f)
