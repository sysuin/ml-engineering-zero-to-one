# Resampling by month, gaps between orders, and time zones.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
tickets = pd.read_sql("SELECT t.opened_at, r.name AS region"
                      " FROM tickets t"
                      " JOIN accounts a USING (account_id)"
                      " JOIN regions r USING (region_id)", con,
                      parse_dates=["opened_at"])
orders = pd.read_sql("SELECT account_id, order_date FROM orders",
                     con, parse_dates=["order_date"])
con.close()

# 1. Resample: GROUP BY month, labelled by a date, empty months kept.
by_time = tickets.set_index("opened_at")
per_month = by_time.resample("MS").size()
print(per_month.loc["2024-04":"2024-06"].to_string())
month_end = by_time.resample("ME").size()
print("June's label with 'ME' instead:",
      month_end.loc["2024-06"].index[0].date())
print()

# 2. Each order's gap: LAG() OVER (PARTITION BY account ORDER BY date).
orders = orders.sort_values(["account_id", "order_date"])
gap = orders.groupby("account_id")["order_date"].diff()
orders["gap_days"] = gap.dt.days
one = orders[(orders["account_id"] == 62)
             & (orders["order_date"] >= "2024-07-01")]
print(one.head(5).to_string(index=False))
print()

# 3. Suppose the West's timestamps were Pacific time. What moves?
west = tickets.loc[tickets["region"] == "West", "opened_at"]
west = west.sort_values()
utc = west.dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")
new_day = utc.dt.date != west.dt.date
new_month = utc.dt.strftime("%Y-%m") != west.dt.strftime("%Y-%m")
print(f"West tickets {len(west):,}: {new_day.sum():,} change date"
      f" in UTC ({new_day.mean():.0%}), {new_month.sum()} change month")
example = new_month & (west >= "2024-01-01")
print("for example", west[example].iloc[0], "Pacific =",
      utc[example].iloc[0].tz_localize(None), "UTC")
