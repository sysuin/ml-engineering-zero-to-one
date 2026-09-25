# LightGBM on the same table: out of the box, too long, and matched.
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC, SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       at_capacity, features, load,
                                       log_loss, top_of_each_cohort)

train, valid = load(*TRAIN), load(*VALIDATION)
X, V = features(train), features(valid)
y, yv = train.not_renewed, valid.not_renewed.to_numpy()

MATCHED = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
               num_leaves=8,              # all a depth-3 tree can have
               min_child_samples=1, min_child_weight=0, reg_lambda=0,
               max_bin=1023)              # scikit-learn's freedoms
models = {
    "logistic (Ch. 7)": RenewalRisk().fit(train),
    "LightGBM, defaults": LGBMClassifier(**LIGHTGBM_DETERMINISTIC),
    "  1,000 rounds": LGBMClassifier(**LIGHTGBM_DETERMINISTIC,
                                     n_estimators=1000),
    "LightGBM, matched": LGBMClassifier(**LIGHTGBM_DETERMINISTIC,
                                        **MATCHED),
    "scikit-learn": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=SEED),
}
scores = {}
print(f"{'validation':<20}{'AUC':>7}{'log loss':>10}"
      f"{'top-40 leavers':>16}")
for name, m in models.items():
    if name.startswith("logistic"):
        p = m.predict_proba(valid)
    else:
        p = m.fit(X, y).predict_proba(V)[:, 1]
    scores[name] = p
    print(f"{name:<20}{auc(yv, p):>7.3f}{log_loss(yv, p):>10.4f}"
          f"{at_capacity(valid, p)['leavers']:>16}")

a, b = scores["LightGBM, matched"], scores["scikit-learn"]
calls = [set(top_of_each_cohort(valid, s).contract_id) for s in (a, b)]
print(f"\nMatched LightGBM against scikit-learn: correlation "
      f"{np.corrcoef(a, b)[0, 1]:.3f},")
print(f"  largest gap {abs(a - b).max():.3f}, "
      f"{len(calls[0] & calls[1])} of 240 calls in common")
