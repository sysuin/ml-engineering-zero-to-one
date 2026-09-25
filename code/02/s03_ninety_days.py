# Exercise 3: "tickets in the 90 days to 31 March 2025", four ways.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
t = pd.read_sql("SELECT opened_at FROM tickets", con,
                parse_dates=["opened_at"])["opened_at"]
con.close()

last_day = pd.Timestamp("2025-03-31")
start = last_day - pd.Timedelta(days=89)   # 90 calendar days in all
end = last_day + pd.Timedelta(days=1)      # first moment NOT included
ninety_back = last_day - pd.Timedelta(days=90)

ways = {
    "between(start, last_day)": t.between(start, last_day).sum(),
    "between(last_day - 90d, last_day)":
        t.between(ninety_back, last_day).sum(),
    "half-open [start, end)": ((t >= start) & (t < end)).sum(),
    "dates in the calendar window":
        t.dt.normalize().between(start, last_day).sum(),
}
print(f"start {start.date()}, end (excluded) {end.date()}, "
      f"{(end - start).days} days")
for how, n in ways.items():
    print(f"  {how:36} {n:>5}")
