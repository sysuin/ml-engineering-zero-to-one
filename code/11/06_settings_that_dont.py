# The settings that barely matter here, each moved from its default
# with the tree size fixed at the choice of the last listing.
import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, load

fit, watch = watch_split(load(*TRAIN))
CHOSEN = dict(learning_rate=0.03, num_leaves=2, min_child_samples=50)


def stopped(**settings):
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       **(CHOSEN | settings))
    m.fit(columns(fit), fit.not_renewed,
          eval_X=columns(watch), eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    loss = m.evals_result_["valid_0"]["binary_logloss"]
    return m.best_iteration_, loss[m.best_iteration_ - 1]


changes = {
    "none (the choice)": {},
    "80% of rows a tree": dict(subsample=0.8, subsample_freq=1),
    "80% of columns a tree": dict(colsample_bytree=0.8),
    "L2 penalty 1 on leaves": dict(reg_lambda=1.0),
    "L2 penalty 10 on leaves": dict(reg_lambda=10.0),
    "63 bins, not 255": dict(max_bin=63),
    "1,023 bins": dict(max_bin=1023),
    "minimum gain 0.01": dict(min_split_gain=0.01),
}
base = stopped()[1]
print(f"{'changed':<26}{'rounds':>7}{'watched loss':>14}{'change':>10}")
for name, settings in changes.items():
    r, loss = stopped(**settings)
    print(f"{name:<26}{r:>7,}{loss:>14.5f}{loss - base:>+10.5f}")
