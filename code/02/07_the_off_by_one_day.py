# June's tickets counted four ways, and what "the last 90 days" means.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
in_sql = con.execute(
    "SELECT COUNT(*) FROM tickets WHERE opened_at"
    " BETWEEN '2024-06-01' AND '2024-06-30'").fetchone()[0]
tickets = pd.read_sql("SELECT ticket_id, account_id, opened_at"
                      " FROM tickets", con)
orders = pd.read_sql("SELECT account_id, order_date FROM orders", con)
con.close()

raw = tickets["opened_at"]
print("opened_at as read:  ", raw.dtype, repr(raw.iloc[0]))
tickets["opened_at"] = pd.to_datetime(raw)
parsed = tickets["opened_at"]
print("after to_datetime:  ", parsed.dtype, parsed.iloc[0])
print()

t = tickets["opened_at"]
counts = {
    "SQL BETWEEN '06-01' AND '06-30'": in_sql,
    "pandas between(06-01, 06-30)":
        t.between("2024-06-01", "2024-06-30").sum(),
    "half-open: >= 06-01 and < 07-01":
        ((t >= "2024-06-01") & (t < "2024-07-01")).sum(),
    "to_period('M') == 2024-06":
        (t.dt.to_period("M") == "2024-06").sum(),
}
for how, n in counts.items():
    print(f"  {how:33} {n:>4} June tickets")
late = t[(t >= "2024-06-30") & (t < "2024-07-01")]
print(f"  opened on 30 June itself: {len(late)},"
      f" the first at {late.min():%H:%M}")
print()

# "The 90 days before 1 August 2024", for one account's orders.
orders["order_date"] = pd.to_datetime(orders["order_date"])
d = orders.loc[orders["account_id"] == 11, "order_date"]
cut = pd.Timestamp("2024-08-01")
start = cut - pd.Timedelta(days=90)
closed = d.between(start, cut)             # both ends included
half_open = (d >= start) & (d < cut)       # start in, cut out
print(f"window from {start:%Y-%m-%d} to {cut:%Y-%m-%d}")
print(f"  both ends included: {len(pd.date_range(start, cut))} days, "
      f"{closed.sum()} orders")
days = len(pd.date_range(start, cut, inclusive="left"))
print(f"  half-open:          {days} days, {half_open.sum()} orders")
print(f"  orders placed on 1 August itself: {(d == cut).sum()}")
