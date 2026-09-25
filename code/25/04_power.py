# Sizing the retention test before any call is made: Appendix F's
# arithmetic with v0.6's list, three designs, and power by simulation.
import json

import numpy as np

from foresight.decide import COSTS
from foresight.impact.analyse import (cohorts_needed, power,
                                      sample_size, simulated_power)
from foresight.impact.lists import lists, scored
from foresight.models.logistic import (TRAIN, at_capacity,
                                       days_since_rule, load)

train = load(*TRAIN)
rule = at_capacity(train, days_since_rule(train))["precision"]
valid = lists(scored("2024-07-01", "2024-12-31"), k=80)
top40 = valid[valid["rank"] <= 40]
p40, p80 = top40.not_renewed.mean(), valid.not_renewed.mean()
save = 0.25                                   # the brief's assumption
print(f"Appendix F, the rule's {rule:.1%} on training:"
      f" {sample_size(rule, rule * (1 - save)):,.0f} per arm")
print(f"v0.6's validation lists: {p40:.1%} of top 40s left,"
      f" {p80:.1%} of top 80s")
print(f"v0.6 at {p40:.1%}:"
      f" {sample_size(p40, p40 * (1 - save)):,.0f} per arm\n")

# Leavers a cohort's calls no longer reach, against calling the top
# 40, and what they are worth if a call saves one in four.
print(f"{'each cohort':<21}{'held':>5}{'called':>7}{'cohorts':>8}"
      f"{'unreached':>10}{'cost':>10}")
designs = [("top 40, 20 held out", p40, 20, 20, 20 * p40),
           ("top 40, 10 held out", p40, 10, 30, 10 * p40),
           ("top 80, 40 held out", p80, 40, 40, 40 * (p40 - p80))]
for name, p0, held, called, lost in designs:
    k = np.ceil(cohorts_needed(p0, p0 * (1 - save), held, called))
    cost = k * lost * save * COSTS.value_at_stake
    print(f"{name:<21}{held:>5}{called:>7}{k:>8.0f}{lost:>10.1f}"
          f"{cost:>10,.0f}")

outcomes = np.stack([c.sort_values("rank").not_renewed.to_numpy()
                     for _, c in top40.groupby("moment")])
rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
years = {"1 year": 12, "2 years": 24, "3 years": 36}
print(f"\nPower with 20 of 40 held out, simulated from the six"
      f" lists\n{'save rate':<11}" + "".join(f"{y:>9}" for y in years)
      + f"{'formula, 1 year':>17}")
curve = []
for s in rates:
    row = [simulated_power(outcomes, s, n) for n in years.values()]
    curve.append(row)
    print(f"{s:<11.2f}" + "".join(f"{v:>9.0%}" for v in row)
          + f"{power(p40, p40 * (1 - s), 240):>17.0%}")
with open("code/25/04_power.json", "w") as f:
    json.dump({"p40": p40, "years": list(years), "rates": rates,
               "curve": curve}, f)
