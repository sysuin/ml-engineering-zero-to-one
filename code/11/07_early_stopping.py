# Early stopping: watch a loss the model is not fitted on, and stop
# where it is lowest. Then: which rows may be watched?
import json

import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.evaluate import auc
from foresight.models.boosting import (RenewalBooster, columns,
                                       watch_split)
from foresight.models.logistic import (TRAIN, VALIDATION, at_capacity,
                                       load, log_loss)

train, valid = load(*TRAIN), load(*VALIDATION)
fit, watch = watch_split(train)

curves = {}
for leaves in (2, 31):          # the choice, and LightGBM's default
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       learning_rate=0.03, num_leaves=leaves,
                       min_child_samples=50)
    m.fit(columns(fit), fit.not_renewed,
          eval_X=(columns(fit), columns(watch)),
          eval_y=(fit.not_renewed, watch.not_renewed))
    r = m.evals_result_
    curves[leaves] = {"fitted": r["valid_0"]["binary_logloss"],
                      "watched": r["valid_1"]["binary_logloss"]}
print(f"{'':>6}{'stumps':>18}{'31 leaves':>20}")
print(f"{'round':>6}{'fitted':>9}{'watched':>9}{'fitted':>11}"
      f"{'watched':>9}")
for n in (10, 50, 100, 300, 1000, 3000):
    a, b = curves[2], curves[31]
    print(f"{n:>6,}{a['fitted'][n - 1]:>9.4f}"
          f"{a['watched'][n - 1]:>9.4f}{b['fitted'][n - 1]:>11.4f}"
          f"{b['watched'][n - 1]:>9.4f}")
for leaves, c in curves.items():
    best = min(range(3000), key=c["watched"].__getitem__)
    print(f"{leaves:>2} leaves: lowest watched loss at round"
          f" {best + 1:,}, {c['watched'][best]:.4f}")

# Where to stop: three ways, all scored on the validation cohorts.
booster = RenewalBooster().fit(train)       # watches April-June 2024
peek = LGBMClassifier(**booster.params(3000)).fit(
    columns(train), train.not_renewed,
    eval_X=columns(valid), eval_y=valid.not_renewed,
    callbacks=[lgb.early_stopping(200, verbose=False)])
early = LGBMClassifier(**booster.params(booster.rounds_)).fit(
    columns(fit), fit.not_renewed)          # stopped, not refitted
ways = {"watch Apr-Jun, refit": (booster.rounds_,
                                 booster.predict_proba(valid)),
        "watch Apr-Jun, no refit": (booster.rounds_,
                                    early.predict_proba(columns(valid))
                                    [:, 1]),
        "watch validation itself": (peek.best_iteration_,
                                    peek.predict_proba(columns(valid))
                                    [:, 1])}
print(f"\n{'stopped by':<25}{'rounds':>7}{'AUC':>7}{'log loss':>10}"
      f"{'leavers':>9}")
for name, (rounds, p) in ways.items():
    y = valid.not_renewed.to_numpy()
    hits = at_capacity(valid, p)["leavers"]
    print(f"{name:<25}{rounds:>7,}{auc(y, p):>7.3f}"
          f"{log_loss(y, p):>10.4f}{hits:>9}")

with open("code/11/07_early_stopping.json", "w") as f:
    json.dump({str(k): v for k, v in curves.items()}, f)
