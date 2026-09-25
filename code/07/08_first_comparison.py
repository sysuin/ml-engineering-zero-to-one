# The first honest comparison: the model against the rule, validation.
import json

from foresight.costs import CostMatrix
from foresight.models.logistic import (CALLS, TRAIN, VALIDATION,
                                       RenewalRisk, at_capacity,
                                       compare, days_since_rule, load,
                                       top_of_each_cohort)

result = compare()                    # fits on train, scores validation
print(f"Validation, top {CALLS} of each monthly cohort, pooled")
print(f"  {'':16}{'calls':>6}{'leavers':>9}{'precision':>11}"
      f"{'recall':>8}")
for name, s in result.items():
    n = s["leavers"]
    n = f"{n:.1f}" if isinstance(n, float) else n
    print(f"  {name:16}{s['calls']:>6}{n:>9}"
          f"{s['precision']:>11.1%}{s['recall']:>8.1%}")
gain = result["model"]["leavers"] - result["rule"]["leavers"]
print(f"Model minus rule: {gain:+d} leaver{'s' * (abs(gain) != 1)}"
      " in six months")
costs = CostMatrix(value_at_stake=2854, save_rate=0.25,
                   call_hours=1.5, hour_cost=60)
for name in ("model", "rule"):
    s = result[name]
    net = costs.net_value(s["leavers"], s["calls"] - s["leavers"])
    print(f"  {name} list, net of its calls: ${net:,.0f}")

# The same two lists at other capacities.
train, valid = load(*TRAIN), load(*VALIDATION)
p = RenewalRisk().fit(train).predict_proba(valid)
rule = days_since_rule(valid)
print("\nLeavers reached at other capacities")
print(f"  {'calls a cohort':<16}{'model':>7}{'rule':>6}"
      f"{'model - rule':>14}")
at = {}
for k in (10, 20, 40, 80, 120):
    m, r = (at_capacity(valid, s, k)["leavers"] for s in (p, rule))
    at[k] = (m, r)
    print(f"  {k:<16}{m:>7}{r:>6}{m - r:>+14}")

mine = set(top_of_each_cohort(valid, p).contract_id)
theirs = set(top_of_each_cohort(valid, rule).contract_id)
print(f"\nCalls on both lists: {len(mine & theirs)} of {len(mine)}")

with open("code/07/08_first_comparison.json", "w") as f:
    json.dump({"result": result, "at": at}, f, default=float)
