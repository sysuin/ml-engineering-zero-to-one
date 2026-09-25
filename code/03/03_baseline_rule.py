# The baseline rule, measured: longest gap since the last order, first.
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
CALLS = 40      # ten a week, for the four weeks before notice

# One row per renewal, with the gap as it stood on its mark.
renewals = pd.read_sql_query("""
    SELECT c.contract_id,
           date(c.end_date, '-90 days')              AS moment,
           c.outcome = 'not_renewed'                 AS left_,
           julianday(date(c.end_date, '-90 days'))
             - julianday(MAX(o.order_date))          AS days_since_order
    FROM contracts c
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date < date(c.end_date, '-90 days')  -- strictly
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome IS NOT NULL
    GROUP BY c.contract_id""", con)

cohorts = renewals.moment.nunique()
never = renewals.days_since_order.isna().sum()
print(f"Renewals scored: {len(renewals):,} in {cohorts} cohorts")
print(f"  no order on record before the mark: {never} (ranked last)\n")

# Rank inside each cohort: the longest gap is called first.
ranked = renewals.sort_values(
    ["moment", "days_since_order", "contract_id"],
    ascending=[True, False, True], na_position="last")
ranked["rank"] = ranked.groupby("moment").cumcount() + 1


def reached(k: int) -> int:
    """Leavers among the top k of every cohort, pooled."""
    return int(ranked.loc[ranked["rank"] <= k, "left_"].sum())


base = renewals.left_.mean()
calls = CALLS * cohorts
leavers = int(renewals.left_.sum())
per_cohort = renewals.groupby("moment").left_.sum()
best = sum(min(CALLS, n) for n in per_cohort)
print(f"Top {CALLS} of each cohort: {calls} calls")
print(f"  {'':18}{'leavers reached':>16}{'precision':>11}")
for name, hits in (("random list", base * calls),
                   ("days-since rule", reached(CALLS)),
                   ("a perfect list", best)):
    print(f"  {name:18}{round(hits, 1):>16g}{hits / calls:>11.1%}")
print(f"The rule reaches {reached(CALLS)} of the {leavers} leavers"
      f" ({reached(CALLS) / leavers:.1%})")

flagged = ranked["rank"] <= CALLS
accuracy = (flagged == ranked.left_.astype(bool)).mean()
print(f"Accuracy, calling the top {CALLS} 'will leave': {accuracy:.1%}")

print("\nThe rule's precision at other capacities")
for k in (10, 20, 40, 80):
    print(f"  top {k:>2} a cohort  ({k / 4:>4g} calls a week)"
          f"   {reached(k) / (k * cohorts):6.1%}")

Path("code/03/03_baseline_rule.json").write_text(json.dumps({
    "calls_per_cohort": CALLS, "cohorts": int(cohorts),
    "base_rate": round(float(base), 4),
    "rule_precision": round(reached(CALLS) / calls, 4),
    "perfect_precision": round(best / calls, 4),
    "precision_at": {k: round(reached(k) / (k * cohorts), 4)
                     for k in (10, 20, 40, 80)},
    "leavers": leavers}, indent=1))
