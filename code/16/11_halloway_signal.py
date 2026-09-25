# Halloway Healthcare's last contract: what was on record at its mark,
# what arrived after it, and what the rows a model learned from that
# morning could have taught about an account like it.
import sqlite3
import textwrap

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.features.build import load
from foresight.slices import key_accounts

con = sqlite3.connect(ML_WAREHOUSE)
c = pd.read_sql_query("""SELECT end_date, notice_date FROM contracts
                         WHERE contract_id = 8118""", con).iloc[0]
t = pd.read_sql_query("""SELECT date(opened_at) AS day, category, body
                         FROM tickets WHERE account_id = 1""", con)
t["day"] = pd.to_datetime(t.day)
table = load()                          # Chapter 4's rows + the library
row = table[table.contract_id == 8118].iloc[0]
mark = row.moment
print(f"Mark {mark:%Y-%m-%d}, notice {c.notice_date},"
      f" end {c.end_date}")

kinds = ["Delivery", "Quality", "Billing", "Account", "Returns"]
print(f"\nTickets by quarter{'':5}" + "".join(f"{k:>9}" for k in kinds))
q = t.assign(q=t.day.dt.to_period("Q"))
for quarter, g in q.groupby("q"):
    n = g.category.value_counts()
    print(f"  {str(quarter):<21}"
          + "".join(f"{n.get(k, 0):>9}" for k in kinds))
late = t[(t.day >= mark) & (t.category == "Account")]
for r in late.itertuples():
    print(textwrap.fill(f'{r.day:%Y-%m-%d}: "{r.body}"', 64,
                        initial_indent="  ", subsequent_indent="    "))

known = known_by(table[table.end_date >= HISTORY_FROM], mark)
key = known.account_id.isin(key_accounts())
print(f"\nAt the mark: {len(known):,} outcomes known,"
      f" {known.not_renewed.sum()} leavers")
print(f"  key-account contracts {key.sum():>6}, leavers"
      f" {known[key].not_renewed.sum()}")
print(f"{'':30}{'Halloway':>10}{'long tail':>11}{'largest':>9}")
lt = known[~key]
for col in ["orders_90d", "tickets_90d", "complaints_90d"]:
    print(f"  {col:<28}{row[col]:>10.0f}{lt[col].median():>11.0f}"
          f"{lt[col].max():>9.0f}")
print("  (long tail: median of its contracts; largest: the most)")

print(f"\nComplaints in 90 days{'contracts':>20}{'left':>8}")
for least in (0, 1, 2, 3, int(row.complaints_90d)):
    had = known[known.complaints_90d >= least]
    rate = f"{had.not_renewed.mean():>8.1%}" if len(had) else ""
    print(f"  {least} or more{'':<22}{len(had):>7,}{rate}")
