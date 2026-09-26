# Exercise 3: a label defined inside the observation window flatters
# any score.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
renewals = pd.read_sql_query("""
    SELECT c.contract_id,
           date(c.end_date, '-90 days')              AS moment,
           c.outcome = 'not_renewed'                 AS left_,
           julianday(date(c.end_date, '-90 days'))
             - julianday(MAX(o.order_date))          AS days_since_order
    FROM contracts c
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date < date(c.end_date, '-90 days')
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome IS NOT NULL
    GROUP BY c.contract_id""", con)

# The flattering label: "churned" means no order in the 90 days
# before the moment.
renewals["quiet_90"] = renewals.days_since_order >= 90

ranked = renewals.sort_values(
    ["moment", "days_since_order", "contract_id"],
    ascending=[True, False, True], na_position="last")
share = renewals.quiet_90.mean()
print(f"Renewals with the flattering label: {share:.1%}")
print(f"{'Precision of the days-since rule':<44}top 10   top 40")
for label, name in (("quiet_90", "no order in the 90 days before"),
                    ("left_", "contract not renewed")):
    p10, p40 = (ranked.groupby("moment").head(k)[label].mean()
                for k in (10, 40))
    print(f"  against '{name + chr(39):<32}{p10:>8.1%}{p40:>9.1%}")
