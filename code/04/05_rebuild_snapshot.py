# A snapshot rebuilt from a log: accounts, from account_history.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
history = pd.read_sql_query("""
    SELECT account_id, valid_from, valid_to, segment
    FROM account_history""", con)
current = pd.read_sql_query("""
    SELECT account_id, name, segment, closed_on FROM accounts
    WHERE crm_source = 'Meridian CRM'""", con)


def snapshot(day: str) -> pd.Series:
    """Each open account's segment on `day`, rebuilt from the log."""
    ended = history.valid_to.notna() & (history.valid_to < day)
    live = history[(history.valid_from <= day) & ~ended]
    gone = current.account_id[current.closed_on < day]
    live = live[~live.account_id.isin(gone)]
    return live.set_index("account_id").segment


# Today's snapshot is the row still open in the log.
open_rows = history[history.valid_to.isna()].set_index("account_id")
both = current.join(open_rows.segment.rename("rebuilt"),
                    on="account_id")
agree = both.segment == both.rebuilt
print(f"Accounts in the current table: {len(both):,}")
print(f"  segment matches the rebuilt snapshot  {agree.sum():,}")
cols = ["account_id", "segment", "rebuilt", "closed_on"]
print(both.loc[~agree, cols].to_string(index=False))

print("\nOpen accounts by segment on three days, rebuilt from the log")
days = ["2023-01-01", "2024-07-01", "2025-12-31"]
counts = pd.DataFrame({d: snapshot(d).value_counts() for d in days})
print(counts.fillna(0).astype(int).to_string())
