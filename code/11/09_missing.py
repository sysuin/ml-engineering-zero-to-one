# The legacy contracts' missing discount: fill it, flag it, or let
# LightGBM route the gap itself.
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.models.boosting import columns, watch_split
from foresight.models.logistic import TRAIN, load

train = load(*TRAIN)
fit, watch = watch_split(train)
CHOSEN = dict(learning_rate=0.03, num_leaves=2, min_child_samples=50)
mean = fit.discount_pct.mean()
assert (fit.discount_pct.isna() == (fit.legacy_terms == 1)).all()


def ways(rows):
    X = columns(rows)
    gap, flag = X.discount_pct, "legacy_terms"
    return {"gap kept, flag in": X,
            "gap kept, no flag": X.drop(columns=flag),
            "filled with 0, flag in": X.assign(
                discount_pct=gap.fillna(0)),
            "filled with 0, no flag": X.assign(
                discount_pct=gap.fillna(0)).drop(columns=flag),
            f"filled with {mean:.1f}, no flag": X.assign(
                discount_pct=gap.fillna(mean)).drop(columns=flag)}


print(f"Discount missing on {fit.discount_pct.isna().sum():,} of "
      f"{len(fit):,} fitted contracts,")
print("all of them on legacy terms\n")
print(f"{'':<26}{'rounds':>7}{'watched loss':>14}")
models = {}
for (name, X), W in zip(ways(fit).items(), ways(watch).values()):
    m = LGBMClassifier(**LIGHTGBM_DETERMINISTIC, n_estimators=3000,
                       **CHOSEN)
    m.fit(X, fit.not_renewed, eval_X=W, eval_y=watch.not_renewed,
          callbacks=[lgb.early_stopping(200, verbose=False)])
    b = m.best_iteration_
    models[name] = m
    loss = m.evals_result_["valid_0"]["binary_logloss"][b - 1]
    print(f"{name:<26}{b:>7,}{loss:>14.5f}")

# Inside the model with no flag: where did the gap go?
m = models["gap kept, no flag"]
col = m.feature_name_.index("discount_pct")
trees = [t["tree_structure"] for t in
         m.booster_.dump_model(num_iteration=m.best_iteration_)
         ["tree_info"]]
cuts = [t for t in trees if t.get("split_feature") == col]
low = sum(t["default_left"] for t in cuts)
print(f"\nNo flag: {len(cuts)} of {len(trees):,} stumps ask about the"
      f" discount; {low} send")
print(f"  a missing discount with the low discounts, {len(cuts) - low}"
      " with the high")
X = ways(fit)["gap kept, no flag"]
print(f"{'discount set to':>18}{'average risk':>14}")
for d in (np.nan, 0, 2, 5, 10):
    p = m.predict_proba(X.assign(discount_pct=d),
                        num_iteration=m.best_iteration_)[:, 1]
    label = "missing" if np.isnan(d) else f"{d}%"
    print(f"{label:>18}{p.mean():>14.1%}")
