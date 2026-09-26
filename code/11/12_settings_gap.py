# The tree-size grid again, watching only cohorts whose outcomes were
# known on 2 May 2024, and what the settings it picks do on validation.
# timeout: 600
from functools import partial

import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.leaderboard import paired, point, run
from foresight.models.boosting import (RenewalBooster, columns,
                                       watch_split)
from foresight.models.logistic import TRAIN, VALIDATION, load
from foresight.models.regularised import maker

train = load(*TRAIN)
first = load(*VALIDATION).moment.min()      # the first validation mark
fit, watch = watch_split(train[train.end_date < first])
print(f"Fitting on {len(fit):,} contracts, watching {len(watch):,}"
      f" ({watch.end_date.min():%b}-{watch.end_date.max():%b %Y})")


def stopped(leaves, smallest):
    """The watched log loss at the best round, as in §11.6."""
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       learning_rate=0.03, num_leaves=leaves,
                       min_child_samples=smallest)
    m.fit(columns(fit), fit.not_renewed,
          eval_X=columns(watch), eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.evals_result_["valid_0"]["binary_logloss"][
        m.best_iteration_ - 1]


LEAVES, SMALLEST = (2, 4, 8, 16, 31), (20, 50, 100, 200)
grid = {(n, k): stopped(n, k) for n in LEAVES for k in SMALLEST}
print(f"\n{'leaves':>6}" + "".join(f"{'min ' + str(k):>12}"
                                   for k in SMALLEST))
for n in LEAVES:
    print(f"{n:>6}" + "".join(f"{grid[n, k]:>12.5f}"
                              for k in SMALLEST))
low = min(grid, key=grid.get)
near = [s for s in grid if grid[s] - grid[low] <= 0.0003]
simple = min(near, key=lambda s: (s[0], grid[s]))
print(f"Lowest: {low[0]} leaves, min {low[1]}; fewest leaves within"
      f" 0.0003: {simple[0]}, min {simple[1]}")

board = run({"lasso": maker("l1", 0.002),
             "booster 2/50": RenewalBooster,
             f"booster {low[0]}/{low[1]}": partial(
                 RenewalBooster, leaves=low[0], min_leaf=low[1]),
             f"booster {simple[0]}/{simple[1]}": partial(
                 RenewalBooster, leaves=simple[0], min_leaf=simple[1])})
print(f"\n{'validation, 240 calls':<22}{'leavers':>8}{'AUC':>7}"
      f"{'   AUC against 2/50'}")
for name in board["runs"]:
    hits = round(point(board, "precision", name) * 240)
    line = f"{name:<22}{hits:>8}{point(board, 'auc', name):>7.3f}"
    if name.startswith("booster") and name != "booster 2/50":
        d, lo, hi = paired(board, "auc", name, "booster 2/50")
        line += f"   {d:+.3f} ({lo:+.3f} to {hi:+.3f})"
    print(line)
