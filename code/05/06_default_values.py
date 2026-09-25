# Hunting for values that were filled in rather than measured.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

# 1. Placeholders anywhere in the warehouse's text columns.
PLACEHOLDERS = ["", "unknown", "n/a", "na", "none", "null", "tbc", "-",
                "1900-01-01", "1970-01-01", "9999-12-31"]
con = sqlite3.connect(ML_WAREHOUSE)
found = 0
for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"):
    for _, col, kind, *_ in con.execute(f"PRAGMA table_info({name})"):
        if kind == "TEXT":
            marks = ",".join("?" * len(PLACEHOLDERS))
            (n,) = con.execute(
                f"SELECT COUNT(*) FROM {name} WHERE lower(trim({col}))"
                f" IN ({marks})", PLACEHOLDERS).fetchone()
            found += n
print(f"Placeholder text or sentinel dates in the warehouse: {found}")

# 2. The most common value of every numeric feature.
train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
features = ["days_since_order", "orders_90d", "orders_prev_90d",
            "spend_365", "tickets_90d", "tenure_days", "discount_pct"]
print("\nMost common value      share  distinct")
for col in features:
    counts = train[col].value_counts(normalize=True, dropna=False)
    print(f"  {col:18}{str(counts.index[0]):>8}{counts.iloc[0]:7.1%}"
          f"{train[col].nunique():>9,}")

# 3. Zeros that are not measurements.
key = pd.read_sql_query(
    "SELECT account_id, is_key_account AS key FROM accounts", con)
con.close()
train = train.merge(key, on="account_id")
zero = train[train.spend_365 == 0]
never = zero[zero.days_since_order.isna()]
early = never[(never.key == 1) & (never.moment < "2023-01-01")]
print(f"\nRows with spend_365 = 0: {len(zero)}")
print(f"  last order over a year before: {len(zero) - len(never)}")
print(f"  no order on record at all:     {len(never)}")
print(f"    key accounts marked before 2023-01-01: {len(early)}")
cols = ["orders_90d", "orders_prev_90d", "tickets_90d", "spend_365"]
zeros = (early[cols] == 0).all(axis=1).sum()
print(f"    all four counts zero on {zeros},"
      f" segment missing on {early.segment.isna().sum()}")
rest = train[(train.key == 1) & ~train.index.isin(early.index)]
print(f"  the other {len(rest)} key-account rows: median spend"
      f" {rest.spend_365.median():,.0f}")
print(f"    any of them zero: {(rest.spend_365 == 0).any()}")
