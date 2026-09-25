# The expectations, run on the real table and on three tidied ones.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.data.expectations import (EXPECTATIONS, ExpectationError,
                                         check_table)

table = pd.read_parquet(TABLE)
check_table(table)                       # raises if anything is broken
print(f"The table as built: {len(table):,} rows,"
      f" {len(EXPECTATIONS)} expectations hold")

train = table[table.end_date <= "2024-06-30"]
tidied = {"discount gaps filled with 0":
          train.assign(discount_pct=train.discount_pct.fillna(0)),
          "no-order recency filled with 0":
          train.assign(
              days_since_order=train.days_since_order.fillna(0))}

# Chapter 4's worst-outcome label: did the account ever leave?
con = sqlite3.connect(ML_WAREHOUSE)
ever = pd.read_sql_query("""
    SELECT account_id, MAX(outcome = 'not_renewed') AS ever_left
    FROM contracts GROUP BY account_id""", con)
con.close()
worst = train.merge(ever, on="account_id")
tidied["label from the worst outcome"] = worst.assign(
    not_renewed=worst.ever_left).drop(columns="ever_left")

for name, broken in tidied.items():
    try:
        check_table(broken)
    except ExpectationError as refusal:
        print(f"\n{name}: {refusal}")
rate = worst.ever_left.mean()
print(f"  (its rate on the training rows: {rate:.1%})")
