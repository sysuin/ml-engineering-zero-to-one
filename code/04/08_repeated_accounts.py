# How often an account appears in the table, and how alike its rows are.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE, rng
from foresight.data.build_table import build

train = build("2023-01-01", "2024-06-30")
rows_per = train.account_id.map(train.account_id.value_counts())
print(f"Training table: {len(train):,} rows,"
      f" {train.account_id.nunique():,} accounts")
for n, k in rows_per.value_counts().sort_index().items():
    rows = "rows" if n > 1 else "row "
    print(f"  accounts with {n} {rows}  {k // n:>6,}"
          f"   ({k:,} rows)")

every = build("2023-01-01", "2025-12-31").account_id.value_counts()
print("Whole table, 2023 to 2025: accounts with 1, 2 and 3 rows")
print("  " + ", ".join(f"{every.eq(n).sum():,}" for n in (1, 2, 3)))

# An account's two rows, a year apart, side by side.
twice = train[rows_per == 2].sort_values("moment")
first = twice.groupby("account_id").nth(0).set_index("account_id")
second = (twice.groupby("account_id").nth(1).set_index("account_id")
               .reindex(first.index))
print(f"\nAccounts with two rows: {len(first):,}, a year apart")
for col in ("spend_365", "orders_90d", "tickets_90d",
            "days_since_order"):
    r = first[col].astype(float).corr(second[col].astype(float))
    print(f"  correlation of {col:18}{r:>6.2f}")
known = first.segment.notna()
same = (first.segment == second.segment)[known].mean()
print(f"  same segment both times {same:>21.1%}")
print(f"  first row not renewed   {first.not_renewed.mean():>21.1%}")


# A random 80/20 split: how many test rows have their account in
# training?
def overlap(accounts: pd.Series) -> float:
    test = rng().random(len(accounts)) < 0.2
    return accounts[test].isin(accounts[~test]).mean()


con = sqlite3.connect(ML_WAREHOUSE)
months = pd.date_range("2022-11-01", "2024-04-01", freq="MS")
account_months = pd.concat(
    pd.read_sql_query("""SELECT account_id FROM contracts
                         WHERE start_date <= :d AND end_date >= :d""",
                      con, params={"d": str(m.date())})
    for m in months).account_id.reset_index(drop=True)
print("\nRandom 80/20 split: test rows whose account is in training")
print(f"  one row per renewal         {len(train):>7,} rows"
      f"  {overlap(train.account_id):>6.1%}")
print(f"  one row per account-month   {len(account_months):>7,} rows"
      f"  {overlap(account_months):>6.1%}")
