# Spend is heavily skewed; on a log scale it is not.
import json
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
con = sqlite3.connect(ML_WAREHOUSE)
key = pd.read_sql_query(
    "SELECT account_id, is_key_account FROM accounts", con)
con.close()
train = train.merge(key, on="account_id")

spend = train.spend_365
logged = np.log10(spend[spend > 0])
print(f"Spend: mean {spend.mean():,.0f}, median {spend.median():,.0f},"
      f" skew {spend.skew():.1f}")
print(f"log10 spend: mean {logged.mean():.2f},"
      f" median {logged.median():.2f}, skew {logged.skew():.2f}")
print(f"  ({(spend == 0).sum()} rows of zero spend have no logarithm)")

total = spend.sum()
top = spend.sort_values(ascending=False)
n1 = len(spend) // 100
print(f"\nThe top 1% of rows ({n1}) hold"
      f" {top.iloc[:n1].sum() / total:.1%} of all spend")
for flag, g in train.groupby("is_key_account"):
    name = "key accounts" if flag else "long tail"
    print(f"  {name:13}{len(g):>6,} rows {len(g) / len(train):6.1%}"
          f"  spend {g.spend_365.sum() / total:6.1%}"
          f"  median {g.spend_365.median():>9,.0f}")

# The mean moves with one row; the median does not.
largest = spend.idxmax()
rest = spend.drop(largest)
print(f"\nWithout the largest row ({spend[largest]:,.0f}):"
      f" mean {rest.mean():,.0f}, median {rest.median():,.0f}")

# For the figure: the same values, binned on each scale.
raw, raw_edges = np.histogram(spend, bins=40)
log, log_edges = np.histogram(logged, bins=np.arange(2.0, 7.01, 0.1))
with open("code/05/02_skew_and_logs.json", "w") as f:
    json.dump({"raw": raw.tolist(), "raw_top": float(raw_edges[-1]),
               "log": log.tolist(), "log_edges": [2.0, 7.0],
               "key_log": np.log10(train.spend_365[
                   (train.is_key_account == 1) & (spend > 0)])
               .round(3).tolist()}, f)
