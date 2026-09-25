# The naive join: features from today's account summary, and their leak.
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
CALLS = 40
TODAY = con.execute("SELECT MAX(order_date) FROM orders").fetchone()[0]

renewals = pd.read_sql_query("""
    SELECT contract_id, account_id,
           date(end_date, '-90 days') AS moment,
           outcome = 'not_renewed' AS left_
    FROM contracts
    WHERE end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND outcome IS NOT NULL""", con)

# The table every analytics team keeps: a row per account, as of today.
summary = pd.read_sql_query("""
    SELECT account_id, MAX(order_date) AS last_order,
           julianday(:today) - julianday(MAX(order_date)) AS days_since,
           SUM(order_date >= date(:today, '-90 days')) AS orders_90d
    FROM orders GROUP BY account_id""", con, params={"today": TODAY})
naive = renewals.merge(summary, on="account_id", how="left")

# The same three facts, measured on the morning of each 90-day mark.
pit = renewals.merge(pd.read_sql_query("""
    SELECT c.contract_id, MAX(o.order_date) AS last_order,
           julianday(date(c.end_date, '-90 days'))
             - julianday(MAX(o.order_date)) AS days_since,
           COALESCE(SUM(o.order_date >= date(c.end_date, '-180 days')),
                    0) AS orders_90d
    FROM contracts c
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date < date(c.end_date, '-90 days')
    GROUP BY c.contract_id""", con), on="contract_id")
counts = ["days_since", "orders_90d"]
for t in (naive, pit):
    t[counts] = t[counts].astype("Int64")

row = 7332          # Chapter 3's contract: account 1678, not renewed
print(f"Contract {row}, built two ways (the summary is as of {TODAY})")
print(f"  {'':24}{'point in time':>15}{'naive':>13}")
for col in ("last_order", "days_since", "orders_90d"):
    a = pit.set_index("contract_id").at[row, col]
    b = naive.set_index("contract_id").at[row, col]
    print(f"  {col:24}{a!s:>15}{b!s:>13}")


def top(table):
    """Rank each cohort by the longest gap; leavers in each top 40."""
    ranked = table.sort_values(["moment", "days_since", "contract_id"],
                               ascending=[True, False, True],
                               na_position="last")
    ranked["rank"] = ranked.groupby("moment").cumcount() + 1
    return ranked[ranked["rank"] <= CALLS].groupby("moment").left_.sum()


cohorts = renewals.moment.nunique()
calls = CALLS * cohorts
perfect = renewals.groupby("moment").left_.sum().clip(upper=CALLS)
print(f"\nThe days-since rule, top {CALLS} a cohort: {calls} calls")
print(f"  {'':22}{'leavers reached':>16}{'precision':>11}")
for name, hits in (("point-in-time table", top(pit)),
                   ("naive table", top(naive)),
                   ("a perfect list", perfect)):
    print(f"  {name:22}{hits.sum():>16}{hits.sum() / calls:>11.1%}")

print("\nShare with no order in the '90 days' feature")
for name, t in (("point in time", pit), ("naive", naive)):
    quiet = t.orders_90d.fillna(0).eq(0).groupby(t.left_).mean()
    print(f"  {name:15}leavers {quiet[1]:6.1%}"
          f"   renewers {quiet[0]:6.1%}")

Path("code/04/01_naive_join_leaks.json").write_text(json.dumps({
    "today": TODAY, "cohorts": [str(m) for m in perfect.index],
    "pit": top(pit).tolist(), "naive": top(naive).tolist(),
    "perfect": perfect.tolist(), "calls": CALLS,
    "example": {name: {c: str(t.set_index("contract_id").at[row, c])
                       for c in ("last_order", *counts)}
                for name, t in (("pit", pit), ("naive", naive))}},
    indent=1))
