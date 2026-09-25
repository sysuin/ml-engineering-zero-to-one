# Segment and region three ways, then a category with 5,000 values.
import sqlite3

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC, ML_WAREHOUSE
from foresight.evaluate import auc
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, load

fit, watch = watch_split(load(*TRAIN))
CHOSEN = dict(learning_rate=0.03, num_leaves=2, min_child_samples=50)
postcode = pd.read_sql_query(
    "SELECT account_id, postcode FROM accounts",
    sqlite3.connect(ML_WAREHOUSE))
codes = pd.Categorical(postcode.postcode).categories
postcode = postcode.set_index("account_id").postcode


def one_hot(rows):
    X = columns(rows)
    return pd.get_dummies(X, columns=["segment", "region"], dtype=float)


def as_numbers(rows):                       # codes 0-3 and 0-4
    X = columns(rows)
    for c in ("segment", "region"):
        X[c] = X[c].cat.codes.astype(float)
    return X


def with_postcode(rows):
    X = columns(rows)
    X["postcode"] = pd.Categorical(
        postcode.reindex(rows.account_id).to_numpy(), categories=codes)
    return X


def stopped(make, **settings):
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       **(CHOSEN | settings))
    m.fit(make(fit), fit.not_renewed,
          eval_X=make(watch), eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    b = m.best_iteration_
    p = m.predict_proba(make(watch), num_iteration=b)[:, 1]
    used = dict(zip(m.feature_name_,
                    m.booster_.feature_importance(iteration=b)))
    return (b, m.evals_result_["valid_0"]["binary_logloss"][b - 1],
            auc(watch.not_renewed, p), used)


UNGUARDED = dict(min_data_per_group=1, cat_smooth=0, cat_l2=0,
                 max_cat_threshold=256)
ways = {"categories (native)": (columns, {}),
        "one-hot columns": (one_hot, {}),
        "codes as numbers": (as_numbers, {}),
        "+ postcode, guarded": (with_postcode, {}),
        "+ postcode, unguarded": (with_postcode, UNGUARDED)}
print(f"{postcode.nunique():,} postcodes; the watched months hold "
      f"{len(watch)} contracts\n")
print(f"{'':<23}{'rounds':>7}{'watched loss':>14}{'AUC':>7}"
      f"{'postcode splits':>17}")
for name, (make, settings) in ways.items():
    b, loss, a, used = stopped(make, **settings)
    print(f"{name:<23}{b:>7,}{loss:>14.5f}{a:>7.3f}"
          f"{used.get('postcode', '-'):>17}")
