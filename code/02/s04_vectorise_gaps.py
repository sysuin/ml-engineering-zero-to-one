# Exercise 4: days since each account's previous order, two ways.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
# Every order, not only 2025's: an account's first order of 2025 has
# a previous order, usually in 2024, and its gap is measured from it.
orders = pd.read_sql("SELECT account_id, order_date FROM orders"
                     " ORDER BY account_id, order_date", con,
                     parse_dates=["order_date"])
con.close()

# The loop you would write first: remember the previous row, compare.
gaps, previous = [], None
for account, date in zip(orders["account_id"], orders["order_date"]):
    if previous is not None and previous[0] == account:
        gaps.append((date - previous[1]).days)
    else:
        gaps.append(None)
    previous = (account, date)
looped = pd.Series(gaps, dtype="float64")

# Vectorised: LAG() OVER (PARTITION BY account_id ORDER BY order_date).
by_account = orders.groupby("account_id")["order_date"]
vectorised = by_account.diff().dt.days.reset_index(drop=True)

in_2025 = (orders["order_date"] >= "2025-01-01").to_numpy()
print(f"{in_2025.sum():,} orders in 2025")
print("same gaps, row for row:",
      looped[in_2025].equals(vectorised[in_2025]))
print("orders in 2025 with no previous order:",
      vectorised[in_2025].isna().sum())
print("median gap in days:", vectorised[in_2025].median())
