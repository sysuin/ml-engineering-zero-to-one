# K-fold cross-validation written out, then checked against the library.
import numpy as np
from sklearn.model_selection import KFold

from foresight.config import SEED
from foresight.evaluate import auc
from foresight.models.logistic import (TRAIN, RenewalRisk,
                                       days_since_rule, load)


def k_folds(n: int, k: int, seed: int):
    """Shuffle the row numbers once, deal them into k folds, and let
    each fold take a turn as the rows held out."""
    order = np.random.RandomState(seed).permutation(n)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1                  # the first folds take one more
    start = 0
    for size in sizes:
        held_out = np.sort(order[start:start + size])
        start += size
        yield np.setdiff1d(np.arange(n), held_out), held_out


train = load(*TRAIN)
y = train.not_renewed.to_numpy()
print(f"{'fold':>4}{'fit on':>8}{'held out':>10}{'leavers':>9}"
      f"{'model AUC':>11}{'rule AUC':>10}")
model_auc, rule_auc = [], []
for i, (fit, out) in enumerate(k_folds(len(train), 5, SEED), 1):
    rows = train.iloc[out]
    p = RenewalRisk().fit(train.iloc[fit]).predict_proba(rows)
    model_auc.append(auc(y[out], p))
    rule_auc.append(auc(y[out], days_since_rule(rows)))
    print(f"{i:>4}{len(fit):>8,}{len(out):>10,}{y[out].sum():>9}"
          f"{model_auc[-1]:>11.3f}{rule_auc[-1]:>10.3f}")
for name, s in (("model", model_auc), ("rule", rule_auc)):
    print(f"{name:>5}: mean {np.mean(s):.3f},"
          f" spread (SD) {np.std(s):.3f}")

library = KFold(n_splits=5, shuffle=True, random_state=SEED)
same = all(np.array_equal(a, c) and np.array_equal(b, d)
           for (a, b), (c, d) in zip(k_folds(len(train), 5, SEED),
                                     library.split(train)))
print(f"Same folds as scikit-learn's KFold: {same}")
