# Exercise 2: constrain orders as well as the gap, and measure the cost.
import lightgbm as lgb
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.evaluate import auc
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, at_capacity, load

fit, watch = watch_split(load(*TRAIN))
X, W = columns(fit), columns(watch)
CHOSEN = dict(learning_rate=0.03, num_leaves=2, min_child_samples=50)
rules = {"none": {},
         "gap rising": {"days_since_order": 1},
         "+ orders falling": {"days_since_order": 1, "orders_90d": -1},
         "+ both order counts": {"days_since_order": 1,
                                 "orders_90d": -1,
                                 "orders_prev_90d": -1}}
print(f"{'constraints':<22}{'rounds':>7}{'watched loss':>14}{'AUC':>7}"
      f"{'leavers':>9}")
for name, rule in rules.items():
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       monotone_constraints=[rule.get(c, 0)
                                             for c in X.columns],
                       **CHOSEN)
    m.fit(X, fit.not_renewed, eval_X=W, eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    b = m.best_iteration_
    p = m.predict_proba(W, num_iteration=b)[:, 1]
    loss = m.evals_result_["valid_0"]["binary_logloss"][b - 1]
    print(f"{name:<22}{b:>7,}{loss:>14.5f}"
          f"{auc(watch.not_renewed, p):>7.3f}"
          f"{at_capacity(watch, p)['leavers']:>9}")

# Why orders resist: the share leaving by orders in the last 90 days.
bands = fit.orders_90d.clip(upper=6)
print("\norders, last 90 days  " + "".join(f"{k:>6}" for k in
                                         ("0", "1", "2", "3", "4", "5",
                                          "6+")))
rate = fit.not_renewed.groupby(bands).mean()
print("share that left       " + "".join(f"{v:>6.1%}" for v in rate))
print("contracts             " + "".join(
    f"{v:>6,}" for v in bands.value_counts().sort_index()))
