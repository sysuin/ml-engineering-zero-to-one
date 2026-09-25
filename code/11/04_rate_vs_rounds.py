# Smaller steps need more of them: four learning rates, stopped early.
import json

import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.evaluate import auc
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, load

fit, watch = watch_split(load(*TRAIN))      # 2023; April-June 2024
print(f"Fitting on {len(fit):,} contracts, watching {len(watch):,}"
      f" ({watch.not_renewed.sum()} leavers)\n")

curves = {}
print(f"{'rate':>6}{'leaves':>8}{'best round':>12}{'watched loss':>14}"
      f"{'AUC':>7}")
for leaves in (31, 4):
    for rate in (0.3, 0.1, 0.03, 0.01):
        m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=6000,
                           learning_rate=rate, num_leaves=leaves)
        m.fit(columns(fit), fit.not_renewed,
              eval_X=columns(watch), eval_y=watch.not_renewed,
              callbacks=[lgb.early_stopping(200, verbose=False)])
        best = m.best_iteration_
        loss = m.evals_result_["valid_0"]["binary_logloss"]
        p = m.predict_proba(columns(watch), num_iteration=best)[:, 1]
        print(f"{rate:>6}{leaves:>8}{best:>12,}{loss[best - 1]:>14.4f}"
              f"{auc(watch.not_renewed, p):>7.3f}")
        if leaves == 4:
            curves[rate] = {"best": best, "loss": loss}

with open("code/11/04_rate_vs_rounds.json", "w") as f:
    json.dump(curves, f)
