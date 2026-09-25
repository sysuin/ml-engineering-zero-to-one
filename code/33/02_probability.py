# Probability from counts: leaving, legacy terms, and Bayes' rule.
import numpy as np
import pandas as pd

from foresight.costs import CostMatrix
from foresight.models.logistic import TRAIN, load

train = load(*TRAIN)
terms = train.legacy_terms.map({0: "current", 1: "legacy"})
outcome = train.not_renewed.map({0: "renewed", 1: "left"})
counts = pd.crosstab(terms, outcome, margins=True, margins_name="all")
table = counts[["left", "renewed", "all"]].rename_axis(
    index=None, columns=None)
print(table.to_string() + "\n")

n = counts.loc["all", "all"]
left, legacy = counts.loc["all", "left"], counts.loc["legacy", "all"]
both = counts.loc["legacy", "left"]
p_leave, p_legacy, p_both = left / n, legacy / n, both / n
p_leave_legacy = both / legacy
p_leave_current = (counts.loc["current", "left"]
                   / counts.loc["current", "all"])
p_legacy_leave = both / left
print(f"P(leave)                  {p_leave:.4f}")
print(f"P(legacy)                 {p_legacy:.4f}")
print(f"P(leave and legacy)       {p_both:.4f}"
      f"   if independent {p_leave * p_legacy:.4f}")
print(f"P(leave | legacy)         {p_leave_legacy:.4f}")
print(f"P(leave | current)        {p_leave_current:.4f}")
print(f"P(legacy | leave)         {p_legacy_leave:.4f}")
print("Bayes: P(legacy | leave) x P(leave) / P(legacy) ="
      f" {p_legacy_leave * p_leave / p_legacy:.4f}")


def odds(p):
    return p / (1 - p)


ratio = p_leave_legacy / p_leave_current
odds_ratio = odds(p_leave_legacy) / odds(p_leave_current)
print(f"\nRisk ratio, legacy to current {ratio:.2f}")
print(f"Odds of leaving: legacy {odds(p_leave_legacy):.4f},"
      f" current {odds(p_leave_current):.4f}, ratio {odds_ratio:.2f}")
print(f"Log-odds: legacy {np.log(odds(p_leave_legacy)):.3f},"
      f" current {np.log(odds(p_leave_current)):.3f}")

# Chapter 3's brief: $2,854 at stake, one save in four, a $90 call.
brief = CostMatrix(value_at_stake=2854, save_rate=0.25,
                   call_hours=1.5, hour_cost=60)
for p in (p_leave_current, p_leave_legacy, brief.break_even()):
    value = p * brief.save_value - brief.call_cost
    print(f"A call at p = {p:.3f}: expected value {value:+7.2f}"
          " dollars")
