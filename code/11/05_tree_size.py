# The settings that matter: how big each tree may grow, and how small
# its leaves may be. Chosen on the watched months, not on validation.
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, load

fit, watch = watch_split(load(*TRAIN))


def stopped(**settings):
    """Fit with early stopping; the best round and its watched loss."""
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       learning_rate=0.03, **settings)
    m.fit(columns(fit), fit.not_renewed,
          eval_X=columns(watch), eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    loss = m.evals_result_["valid_0"]["binary_logloss"]
    return m.best_iteration_, loss[m.best_iteration_ - 1]


LEAVES, SMALLEST = (2, 4, 8, 16, 31), (20, 50, 100, 200)
grid = {(n, k): stopped(num_leaves=n, min_child_samples=k)
        for n in LEAVES for k in SMALLEST}
print("Watched log loss at the best round (best round beneath)")
print(f"{'leaves':>6}" + "".join(f"{'min ' + str(k):>12}"
                                 for k in SMALLEST))
for n in LEAVES:
    print(f"{n:>6}" + "".join(f"{grid[n, k][1]:>12.5f}"
                              for k in SMALLEST))
    print(f"{'':>6}" + "".join(f"{grid[n, k][0]:>12,}"
                               for k in SMALLEST))
n, k = min(grid, key=lambda s: grid[s][1])
spread = np.ptp([v[1] for v in grid.values()])
print(f"\nLowest: {n} leaves, at least {k} contracts a leaf,"
      f" {grid[n, k][0]:,} rounds")
print(f"From best to worst of the 20 settings: {spread:.4f}")

print(f"\n{'31 leaves, depth limited to':<28}{'rounds':>7}{'loss':>9}")
for depth in (2, 3, 5):
    r, loss = stopped(num_leaves=31, max_depth=depth)
    print(f"{depth:>28}{r:>7,}{loss:>9.5f}")
