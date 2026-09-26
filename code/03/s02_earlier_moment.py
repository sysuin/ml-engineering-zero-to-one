# Exercise 4: what the rule loses when the list must arrive at 120 days,
# not 90.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)

print("Days-since rule, top 40 a cohort, "
      "renewals ending 2023 to mid-2024")
for lead in (90, 120):
    renewals = pd.read_sql_query(f"""
        SELECT c.contract_id,
               date(c.end_date, '-{lead} days')          AS moment,
               c.outcome = 'not_renewed'                 AS left_,
               julianday(date(c.end_date, '-{lead} days'))
                 - julianday(MAX(o.order_date))
                                                 AS days_since_order
        FROM contracts c
        LEFT JOIN orders o
               ON o.account_id = c.account_id
              AND o.order_date < date(c.end_date, '-{lead} days')
        WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
          AND c.outcome IS NOT NULL
        GROUP BY c.contract_id""", con)
    top = (renewals.sort_values(
               ["moment", "days_since_order", "contract_id"],
               ascending=[True, False, True], na_position="last")
           .groupby("moment").head(40))
    print(f"  scored {lead:>3} days out"
          f"   precision {top.left_.mean():.1%}"
          f"   leavers reached {int(top.left_.sum())}")
