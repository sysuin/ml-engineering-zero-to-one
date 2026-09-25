# The same as-of join in pandas, with merge_asof, checked against SQL.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
renewals = pd.read_sql_query("""
    SELECT contract_id, account_id, date(end_date, '-90 days') AS moment
    FROM contracts
    WHERE end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND outcome IS NOT NULL""", con, parse_dates=["moment"])
orders = pd.read_sql_query(
    "SELECT order_id, account_id, order_date, channel FROM orders",
    con, parse_dates=["order_date"])

# Both sides sorted on the time column; `by` keeps accounts apart.
asof = pd.merge_asof(
    renewals.sort_values("moment"),
    orders.sort_values(["order_date", "order_id"]),
    left_on="moment", right_on="order_date", by="account_id",
    direction="backward",           # the latest order at or before...
    allow_exact_matches=False)      # ...strictly before: not on the day

# The SQL answer from the previous listing, for comparison.
sql = pd.read_sql_query("""
    SELECT c.contract_id, MAX(o.order_date) AS last_order
    FROM contracts c
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date < date(c.end_date, '-90 days')
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome IS NOT NULL
    GROUP BY c.contract_id""", con, parse_dates=["last_order"])


def same(a, b):
    """Equal dates, counting 'no order' on both sides as equal."""
    return (a == b) | (a.isna() & b.isna())


both = asof.merge(sql, on="contract_id")
agree = same(both.order_date, both.last_order)
print(f"merge_asof and SQL agree on {agree.sum():,} of {len(both):,}")

# The default lets an order placed on the day of the moment through.
loose = pd.merge_asof(
    renewals.sort_values("moment"),
    orders.sort_values(["order_date", "order_id"]),
    left_on="moment", right_on="order_date", by="account_id")
moved = ~same(loose.set_index("contract_id").order_date,
              asof.set_index("contract_id").order_date)
print("With allow_exact_matches left at its default,")
print(f"{moved.sum()} rows take an order placed on their moment's day")
