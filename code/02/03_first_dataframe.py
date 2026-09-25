# Meridian's accounts as a DataFrame: shape, index, columns, dtypes.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
accounts = pd.read_sql(
    "SELECT account_id, name, region_id, segment, since,"
    " is_key_account FROM accounts",
    con,
    index_col="account_id",         # the key becomes the row labels
)
con.close()

print("shape:", accounts.shape)    # (rows, columns)
print()
print(accounts[["name", "segment", "since"]].head(4).to_string())
print()
print(accounts.dtypes.to_string())
print()

since = accounts["since"]           # one column: a Series
print(type(since).__name__, "of", since.dtype,
      "- first value", repr(since.iloc[0]))
accounts["since"] = pd.to_datetime(accounts["since"])
print("after to_datetime:", accounts["since"].dtype)
print()
print(accounts.loc[11, ["name", "segment", "since"]].to_string())
