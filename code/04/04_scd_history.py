# Segment as it was at each moment, and the segment the CRM holds today.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
history = pd.read_sql_query("""
    SELECT account_id, valid_from, valid_to, segment
    FROM account_history""", con,
    parse_dates=["valid_from", "valid_to"])
renewals = pd.read_sql_query("""
    SELECT c.contract_id, c.account_id, date(c.end_date, '-90 days')
             AS moment, c.outcome = 'not_renewed' AS left_,
           a.segment AS segment_today
    FROM contracts c JOIN accounts a USING (account_id)
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome IS NOT NULL""", con, parse_dates=["moment"])

print("account_history for account 276, Hartley Salon")
print(history[history.account_id == 276].to_string(index=False))

# The row in force the day before the moment: the latest valid_from that
# is strictly earlier than the moment.
rows = pd.merge_asof(renewals.sort_values("moment"),
                     history.sort_values("valid_from"),
                     left_on="moment", right_on="valid_from",
                     by="account_id", allow_exact_matches=False)
rows = rows.rename(columns={"segment": "segment_then"})
print("\nIts renewals, both ways")
cols = ["contract_id", "moment", "segment_then", "segment_today"]
print(rows.loc[rows.account_id == 276, cols].to_string(index=False))

missing = rows.segment_then.isna()
differ = ~missing & (rows.segment_then != rows.segment_today)
accounts = rows.account_id[differ].nunique()
print(f"\nRenewals {len(rows):,}")
print(f"  no history row yet at the moment      {missing.sum():>5}")
print(f"  today's segment is not the one then   {differ.sum():>5}"
      f"  ({accounts} accounts)")
print(rows[differ].groupby(["segment_then", "segment_today"])
      .size().to_string())
left = rows.left_[differ]
print(f"Not renewed, where the two differ   {left.mean():6.1%}"
      f"  ({left.sum()} of {len(left)})")
print(f"Not renewed, all renewals           {rows.left_.mean():6.1%}")
