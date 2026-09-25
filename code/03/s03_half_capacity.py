# Exercise 5: half the team, and the save rate at which the rule stops paying.
import json
from pathlib import Path

from foresight.costs import CostMatrix

# The inputs and precisions that 04_cost_matrix.py and 03_baseline_rule.py saved.
costs = json.loads(Path("code/03/04_cost_matrix.json").read_text())
rule = json.loads(Path("code/03/03_baseline_rule.json").read_text())
m = CostMatrix(value_at_stake=costs["value_at_stake"], save_rate=costs["save_rate"],
               call_hours=costs["call_hours"], hour_cost=costs["hour_cost"])

for calls in (40, 20):
    precision = rule["precision_at"][str(calls)]
    each = m.value_per_call(precision)
    print(f"{calls} calls a cohort: {each:+,.0f} a call,"
          f" {each * calls:+,.0f} a month,"
          f" {calls * m.call_hours:g} hours")

# The save rate at which a call from the top 40 only just pays for itself.
breakeven = m.call_cost / (rule["precision_at"]["40"] * m.value_at_stake)
print(f"The rule's list stops paying below a save rate of {breakeven:.1%}")
