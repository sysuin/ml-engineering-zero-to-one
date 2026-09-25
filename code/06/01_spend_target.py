# The number to predict: what each account spent after its 90-day mark.
import sqlite3

from foresight.config import ML_WAREHOUSE
from foresight.data.spend import TRAIN, VALIDATION, spend_table

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
print(f"training rows   {len(train):>6,}  ending "
      f"{TRAIN[0]} to {TRAIN[1]}")
print(f"validation rows {len(valid):>6,}  ending "
      f"{VALIDATION[0]} to {VALIDATION[1]}")
days = (train.end_date - train.moment).dt.days
print(f"days from mark to end: {days.min()} to {days.max()}")

row = train[train.contract_id == 7332].iloc[0]
print(f"\ncontract 7332: mark {row.moment:%Y-%m-%d},"
      f" ends {row.end_date:%Y-%m-%d}")
print(f"  spend_90d        {row.spend_90d:>12,.2f}  before the mark")
print(f"  spend_next_90d   {row.spend_next_90d:>12,.2f}  the target")

y = train.spend_next_90d
print("\nThe target on the training rows, in dollars")
print(f"  mean {y.mean():>9,.0f}      median {y.median():>9,.0f}")
for q in (0.25, 0.75, 0.99):
    print(f"  {q:.0%} of rows spent less than {y.quantile(q):>9,.0f}")
top = y.nlargest(len(y) // 100)
share = top.sum() / y.sum()
print(f"  top 1% of rows ({len(top)}) spent {share:.1%} of all of it")
big = train.loc[y.idxmax()]
with sqlite3.connect(ML_WAREHOUSE) as con:
    (name,) = con.execute("SELECT name FROM accounts WHERE account_id"
                          " = ?", (int(big.account_id),)).fetchone()
print(f"  the largest row: {name}, {big.spend_next_90d:,.0f}")

print("\nSpend after the mark, against the 90 days before it")
print(f"  {'':<10}{'rows':>7}{'median ratio':>15}{'spent nothing':>15}")
for label, group in train.groupby("not_renewed"):
    had = group[group.spend_90d > 0]
    ratio = (had.spend_next_90d / had.spend_90d).median()
    none = (group.spend_next_90d == 0).mean()
    name = "leavers" if label else "renewers"
    print(f"  {name:<10}{len(group):>7,}{ratio:>15.2f}{none:>15.1%}")
