# Four pairs of Meridian columns with nearly the same correlation.
import json
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
con = sqlite3.connect(ML_WAREHOUSE)
accounts = pd.read_sql_query(
    "SELECT account_id, name, is_key_account AS key FROM accounts", con)
con.close()
train = train.merge(accounts, on="account_id")
train["days"] = train.days_since_order.astype(float)
train["log_days"] = np.log1p(train.days)       # log(1 + days)

tail = train[train.key == 0]
panels = {
    "A long tail": (tail, "orders_90d", "spend_365"),
    "B with an order": (train.dropna(subset=["days"]),
                        "days", "log_days"),
    "C mid-market": (train[train.segment == "Mid-market"],
                     "orders_90d", "orders_prev_90d"),
    "D key accounts": (train[train.key == 1],
                       "orders_prev_90d", "tickets_90d"),
}
print(f"{'':15}{'x':>16}{'y':>16}{'rows':>6}{'r':>7}{'rank r':>7}")
saved = {}
for name, (df, x, y) in panels.items():
    r = df[x].corr(df[y])
    rank = df[x].corr(df[y], method="spearman")
    print(f"{name:15}{x:>16}{y:>16}{len(df):>6,}{r:>7.2f}{rank:>7.2f}")
    saved[name] = {"x": x, "y": y, "r": round(r, 2),
                   "points": df[[x, y]].round(3).values.tolist()}

# What each r rests on.
c = panels["C mid-market"][0]
lt = c[c.key == 0]
saved["C mid-market"]["without"] = round(
    lt.orders_90d.corr(lt.orders_prev_90d), 2)
print(f"\nC without its {c.key.sum()} key accounts:"
      f" r = {saved['C mid-market']['without']:.2f}")
d = panels["D key accounts"][0]
no_h = d[~d.name.str.startswith("Halloway")]
saved["D key accounts"]["without"] = round(
    no_h.orders_prev_90d.corr(no_h.tickets_90d), 2)
print(f"D without Halloway's {len(d) - len(no_h)} rows:"
      f" r = {saved['D key accounts']['without']:.2f}")
print("B is exact: log_days is a function of days")

with open("code/05/07_anscombe_meridian.json", "w") as f:
    json.dump(saved, f)
