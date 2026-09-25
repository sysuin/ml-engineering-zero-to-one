# "Spend in the year before the mark", cohort by cohort.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
con = sqlite3.connect(ML_WAREHOUSE)
key = pd.read_sql_query(
    "SELECT account_id, is_key_account AS key FROM accounts", con)
con.close()
train = train.merge(key, on="account_id")

# Where each group's order history begins in the warehouse.
starts = {0: pd.Timestamp("2022-01-01"), 1: pd.Timestamp("2023-01-01")}


def covered(moment, start):
    """Days of the 365-day window that fall inside the record."""
    return max(0, min(365, (moment - start).days))


print(f"{'mark':12}{'long tail':>22}{'key accounts':>28}")
print(f"{'':12}{'days':>8}{'median spend':>14}"
      f"{'rows':>6}{'days':>8}{'median spend':>14}")
for moment, g in train.groupby("moment"):
    lt, ka = g[g.key == 0], g[g.key == 1]
    print(f"{moment:%Y-%m-%d}  {covered(moment, starts[0]):>8}"
          f"{lt.spend_365.median():>14,.0f}{len(ka):>6}"
          f"{covered(moment, starts[1]):>8}"
          f"{ka.spend_365.median():>14,.0f}")
