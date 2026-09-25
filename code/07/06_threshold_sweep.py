# One model, six thresholds: who gets called, and what it is worth.
import json


from foresight.costs import CostMatrix
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       load, top_of_each_cohort)

train, valid = load(*TRAIN), load(*VALIDATION)
p = RenewalRisk().fit(train).predict_proba(valid)
y = valid.not_renewed.to_numpy()
months = valid.moment.nunique()
# The brief's figures: $2,854 at stake, one save in four, a $90 call.
costs = CostMatrix(value_at_stake=2854, save_rate=0.25,
                   call_hours=1.5, hour_cost=60)

print(f"Validation: {len(y):,} contracts, {y.sum()} leavers,"
      f" {months} cohorts")
print(f"  {'threshold':>9}{'called':>8}{'a month':>9}{'reached':>9}"
      f"{'precision':>11}{'recall':>8}{'net $':>9}")
for t in (0.5, 0.3, 0.2, costs.break_even(), 0.1, 0.05):
    called = p >= t
    n = int(called.sum())
    tp = int((called & (y == 1)).sum())
    print(f"  {t:>9.1%}{n:>8}{n / months:>9.0f}{tp:>9}"
          f"{tp / n:>11.1%}{tp / y.sum():>8.1%}"
          f"{costs.net_value(tp, n - tp):>+9,.0f}")

# Capacity sets its own threshold: the 40th score of each cohort.
top = top_of_each_cohort(valid, p)
floor = top.groupby("moment").score.min()
print("\nThe 40th score in each cohort, the threshold capacity sets")
print("  " + "  ".join(f"{m:%b} {s:.1%}" for m, s in floor.items()))

with open("code/07/06_threshold_sweep.json", "w") as f:
    json.dump({"left": p[y == 1].tolist(), "stayed": p[y == 0].tolist(),
               "break_even": costs.break_even(),
               "floors": floor.tolist()}, f)
