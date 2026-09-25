# Exercise 2: what a 90%-accurate list looks like on validation.
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       load)

train, valid = load(*TRAIN), load(*VALIDATION)
p = RenewalRisk().fit(train).predict_proba(valid)
y = valid.not_renewed.to_numpy()
print(f"  {'threshold':>9}{'accuracy':>10}{'called':>8}{'leavers':>9}"
      f"{'recall':>8}")
for t in (1.01, 0.5, 0.3, 0.2, 0.15, 0.1):
    called = p >= t
    tp = int((called & (y == 1)).sum())
    label = "nobody" if t > 1 else f"{t:.0%}"
    print(f"  {label:>9}{(called == (y == 1)).mean():>10.1%}"
          f"{called.sum():>8}{tp:>9}{tp / y.sum():>8.1%}")
