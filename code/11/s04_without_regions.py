# Exercise 4: the booster with and without the region, on the watched
# months, and how often its stumps asked about the region at all.
import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.evaluate import auc
from foresight.models.boosting import (RenewalBooster, columns,
                                       watch_split)
from foresight.models.logistic import TRAIN, at_capacity, load

fit, watch = watch_split(load(*TRAIN))
params = RenewalBooster().params(3000)
print(f"{'':<18}{'rounds':>7}{'watched loss':>14}{'AUC':>7}"
      f"{'leavers':>9}{'region stumps':>15}")
for name, drop in (("with region", []), ("without region", ["region"])):
    X, W = columns(fit).drop(columns=drop), columns(watch).drop(
        columns=drop)
    rising = [int(c == "days_since_order") for c in X.columns]
    m = LGBMClassifier(**(params | {"monotone_constraints": rising}))
    m.fit(X, fit.not_renewed, eval_X=W, eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    b = m.best_iteration_
    p = m.predict_proba(W, num_iteration=b)[:, 1]
    used = dict(zip(m.feature_name_,
                    m.booster_.feature_importance(iteration=b)))
    loss = m.evals_result_["valid_0"]["binary_logloss"][b - 1]
    print(f"{name:<18}{b:>7,}{loss:>14.5f}"
          f"{auc(watch.not_renewed, p):>7.3f}"
          f"{at_capacity(watch, p)['leavers']:>9}"
          f"{used.get('region', 0):>15}")
