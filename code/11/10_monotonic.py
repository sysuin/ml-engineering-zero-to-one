# A monotonic constraint: a longer gap since the last order may never
# lower the risk. What it changes, and what it costs.
import json

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.evaluate import auc
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, at_capacity, load

fit, watch = watch_split(load(*TRAIN))
CHOSEN = dict(learning_rate=0.03, num_leaves=2, min_child_samples=50)
X, W = columns(fit), columns(watch)
RISING = [int(c == "days_since_order") for c in X.columns]
GAPS = np.arange(0, 366, 5)

print(f"{'':<14}{'rounds':>7}{'watched loss':>14}{'AUC':>7}"
      f"{'leavers':>9}{'falls':>7}")
shape = {}
for name, extra in (("unconstrained", {}),
                    ("constrained", {"monotone_constraints": RISING})):
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       **CHOSEN, **extra)
    m.fit(X, fit.not_renewed, eval_X=W, eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    b = m.best_iteration_
    p = m.predict_proba(W, num_iteration=b)[:, 1]
    # The partial response: every fitted contract given the same gap,
    # the rest of its columns as they are, and the risks averaged.
    shape[name] = [float(m.predict_proba(
        X.assign(days_since_order=float(g)), num_iteration=b)[:, 1]
        .mean()) for g in GAPS]
    falls = int((np.diff(shape[name]) < 0).sum())
    loss = m.evals_result_["valid_0"]["binary_logloss"][b - 1]
    print(f"{name:<14}{b:>7,}{loss:>14.5f}"
          f"{auc(watch.not_renewed, p):>7.3f}"
          f"{at_capacity(watch, p, 40)['leavers']:>9}{falls:>7}")

print(f"\n{'days since order':>16}{'unconstrained':>15}"
      f"{'constrained':>13}")
for g in (0, 30, 60, 90, 120, 180, 240, 270, 300, 365):
    i = int(np.where(GAPS == g)[0][0]) if g < 365 else -1
    print(f"{g:>16}{shape['unconstrained'][i]:>15.1%}"
          f"{shape['constrained'][i]:>13.1%}")

drop = int(GAPS[1:][np.diff(shape["unconstrained"]) < 0][0])
print(f"\nThe unconstrained risk falls at a gap of {drop} days.")
gap, y = fit.days_since_order, fit.not_renewed
bands = [(0, 59), (60, 119), (120, 179), (180, 239), (240, drop - 1),
         (drop, 365)]
print(f"{'fitted contracts':>16}{'contracts':>11}{'left':>7}")
for lo, hi in bands[3:]:
    b = gap.between(lo, hi)
    print(f"{f'{lo} to {hi} days':>16}{b.sum():>11}{y[b].mean():>7.1%}")

observed = [[lo, hi, float(y[gap.between(lo, hi)].mean())]
            for lo, hi in bands]
with open("code/11/10_monotonic.json", "w") as f:
    json.dump({"gaps": GAPS.tolist(), **shape, "bands": observed}, f)
