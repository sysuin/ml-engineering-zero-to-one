# Exercise 2: every account's first order on record, week by week.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
first = pd.read_sql_query("""
    SELECT a.account_id, a.is_key_account AS key, a.since,
           MIN(o.order_date) AS first_order
    FROM accounts a JOIN orders o USING (account_id)
    WHERE a.crm_source = 'Meridian CRM'
    GROUP BY a.account_id""", con, parse_dates=["since", "first_order"])
con.close()
week = first.first_order.dt.to_period("W").astype(str)
top = week.value_counts().head(4)
print(f"{len(first):,} accounts with an order;"
      " the busiest weeks for a first order")
for w, n in top.items():
    print(f"  {w}  {n:>5,}")
print(f"A typical week after 2022: {week.value_counts().median():.0f}")
for flag, g in first.groupby("key"):
    start = "2023-01-01" if flag else "2022-01-01"
    early = g.since < start
    month_one = pd.Timestamp(start) + pd.Timedelta(days=31)
    at_start = g.first_order < month_one
    name = "key accounts" if flag else "long tail"
    print(f"{name}: {early.sum():,} customers from before {start}")
    print(f"  {(early & at_start).sum():,} of them 'first ordered' in"
          " the record's first month")
