# An as-of join in SQL: each contract's last order before its moment.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
RENEWALS = """
    SELECT contract_id, account_id, date(end_date, '-90 days') AS moment
    FROM contracts
    WHERE end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND outcome IS NOT NULL"""

# One: a correlated subquery. Good for one column of the matching row.
subquery = pd.read_sql_query(f"""
    SELECT r.contract_id,
           (SELECT MAX(o.order_date) FROM orders o
            WHERE o.account_id = r.account_id
              AND o.order_date < r.moment) AS last_order
    FROM ({RENEWALS}) r""", con)

# Two: a window function. Numbers the matches, newest first, and keeps
# the first, so every column of the matching row comes with it.
window = pd.read_sql_query(f"""
    WITH matches AS (
        SELECT r.contract_id, o.order_id, o.order_date, o.channel,
               ROW_NUMBER() OVER (PARTITION BY r.contract_id
                                  ORDER BY o.order_date DESC,
                                           o.order_id DESC) AS n
        FROM ({RENEWALS}) r
        JOIN orders o
          ON o.account_id = r.account_id
         AND o.order_date < r.moment)
    SELECT contract_id, order_id, order_date AS last_order, channel
    FROM matches WHERE n = 1""", con)

both = subquery.merge(window, on="contract_id", how="left",
                      suffixes=("", "_w"))
found = both.last_order.notna().sum()
agree = (both.last_order.fillna("none")
         == both.last_order_w.fillna("none"))
print(f"Renewals: {len(both):,}")
print(f"  with an order before the moment   {found:>6,}")
print(f"  both queries give the same date    {agree.sum():>6,}")
print("\nThe window query's row for Chapter 3's contract")
print(window[window.contract_id == 7332].to_string(index=False))
