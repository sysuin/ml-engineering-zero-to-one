# Exercise 2: 2025 revenue from accounts with a 2025 ticket.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
sales = pd.read_sql("SELECT account_id, revenue FROM v_sales"
                    " WHERE order_date >= '2025-01-01'", con)
tickets = pd.read_sql("SELECT ticket_id, account_id FROM tickets"
                      " WHERE opened_at >= '2025-01-01'", con)
con.close()


def report(label, frame):
    total = frame["revenue"].sum()
    print(f"{label:18}{total:>16,.2f} ({len(frame):,} rows)")


wrong = sales.merge(tickets, on="account_id")   # line x ticket pairs
report("colleague's total", wrong)

with_ticket = tickets["account_id"].unique()    # the key, made unique
right = sales[sales["account_id"].isin(with_ticket)]   # a filter
report("corrected total", right)
report("all 2025 revenue", sales)

# The same answer as a merge, with the promise written down.
keys = pd.DataFrame({"account_id": with_ticket})
check = sales.merge(keys, on="account_id", validate="many_to_one")
print("merge on unique keys agrees:", round(check["revenue"].sum(), 2)
      == round(right["revenue"].sum(), 2))
