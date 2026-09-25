# The same descent at three learning rates: too small, right, too large.
import json

import numpy as np

from foresight.data.spend import TRAIN, spend_table

train = spend_table(*TRAIN)
x = train.spend_90d.to_numpy() / 1000
y = train.spend_next_90d.to_numpy() / 1000


def run(lr, steps=5000):
    """The loss before every step of a descent from w = b = 0."""
    w, b, losses = 0.0, 0.0, []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps + 1):
            error = w * x + b - y
            losses.append(float(np.mean(error ** 2)))
            w -= lr * 2 * np.mean(error * x)
            b -= lr * 2 * np.mean(error)
    return losses, w, b


rates = {"too small": 0.00002, "right": 0.0015, "too large": 0.002}
checks = (0, 1, 10, 100, 1000, 5000)
print(f"{'loss at step':<20}" + "".join(f"{s:>8}" for s in checks))
runs = {}
for name, lr in rates.items():
    losses, w, b = runs[name] = run(lr)
    cells = "".join(f"{v:>8.1f}" if v < 1e6 else f"{v:>8.0e}"
                    for v in (losses[s] for s in checks))
    print(f"{name:<10}{lr:<10}" + cells.replace("inf", "  inf"))
    print(f"{'':<10}ends at w {w:.4f}, b {b:.4f}")

# The largest rate that does not diverge, from the data's own numbers.
curvature = 2 * np.array([[np.mean(x * x), np.mean(x)],
                          [np.mean(x), 1.0]])
limit = 2 / np.linalg.eigvalsh(curvature).max()
print(f"\ndescent diverges for learning rates above {limit:.6f}")

with open("code/06/05_learning_rates.json", "w") as f:
    json.dump({name: {"lr": rates[name], "loss": r[0][:201]}
               for name, r in runs.items()}, f)
