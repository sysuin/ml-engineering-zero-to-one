# Exercise 2: a colleague's one-query builder, against Foresight's.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import build

con = sqlite3.connect(ML_WAREHOUSE)
theirs = pd.read_sql_query("""
    SELECT c.contract_id, c.account_id,
           date(c.end_date, '-90 days') AS moment,
           (SELECT COUNT(*) FROM tickets t
             WHERE t.account_id = c.account_id
               AND t.opened_at >= date(c.end_date, '-180 days')
               AND t.opened_at < c.end_date) AS tickets_90d,
           a.segment,
           c.outcome = 'not_renewed' AS not_renewed
    FROM contracts c JOIN accounts a USING (account_id)
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome IS NOT NULL""", con).set_index("contract_id")
ours = build("2023-01-01", "2024-06-30").set_index("contract_id")
theirs = theirs.loc[ours.index]

print(f"Rows: theirs {len(theirs):,}, ours {len(ours):,}")
for col in ("tickets_90d", "segment", "not_renewed"):
    a, b = theirs[col], ours[col]
    differ = ~((a == b) | (a.isna() & b.isna()))
    print(f"  {col:12} differs on {differ.sum():>5,} rows")

more = theirs.tickets_90d > ours.tickets_90d
print("\nTickets: rows where theirs counts more, by label")
print(f"  leavers  {more[ours.not_renewed == 1].mean():6.1%}")
print(f"  renewers {more[ours.not_renewed == 0].mean():6.1%}")
print(f"Mean tickets_90d, leavers: theirs "
      f"{theirs.tickets_90d[ours.not_renewed == 1].mean():.2f},"
      f" ours {ours.tickets_90d[ours.not_renewed == 1].mean():.2f}")
