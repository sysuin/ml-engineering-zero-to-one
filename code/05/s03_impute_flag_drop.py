# Exercise 3: the rows missing segment or recency, looked at one by one.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
con = sqlite3.connect(ML_WAREHOUSE)
accounts = pd.read_sql_query(
    "SELECT account_id, is_key_account AS key, segment AS segment_now"
    " FROM accounts", con)
con.close()
gaps = train[train.segment.isna() | train.days_since_order.isna()]
gaps = gaps.merge(accounts, on="account_id")
print(f"{'contract':>8}  {'mark':10} key  {'segment then':15}"
      f"{'segment now':15}{'tenure':>6}{'left':>5}")
for _, r in gaps.iterrows():
    then = r.segment if isinstance(r.segment, str) else "(missing)"
    print(f"{r.contract_id:>8}  {r.moment:%Y-%m-%d} {r.key:>3}  "
          f"{then:15}{r.segment_now:15}{r.tenure_days:>6}"
          f"{r.not_renewed:>5}")
print(f"\n{len(gaps)} rows, not renewed: {gaps.not_renewed.sum()};"
      f" segment missing on {gaps.segment.isna().sum()}")
