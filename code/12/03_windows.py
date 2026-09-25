# Counts and sums over 30, 90 and 365 days, every window ending the day
# before the mark, and what a window that touches the mark would add.
import pandas as pd

from foresight.evaluate import auc
from foresight.features.build import load
from foresight.features.sources import Context, load_sources
from foresight.models.featured import TRAINING

table = load()
train = table[table.end_date.between(*TRAINING)]
y = train.not_renewed

# Which window carries most? Each alone, ranking training contracts.
print(f"{'Training, one column alone':<28}{'30 days':>9}{'90 days':>9}"
      f"{'365 days':>10}")
for what, cols in (("orders", ["orders_30d", "orders_90d",
                               "orders_365"]),
                   ("spend", ["spend_30d", "spend_90d", "spend_365"])):
    a = [1 - auc(y, train[c]) for c in cols]
    print(f"  {what + ', AUC':<26}{a[0]:>9.3f}{a[1]:>9.3f}"
          f"{a[2]:>10.3f}")

# How much the windows repeat each other: rank correlation, orders.
r = train[["orders_30d", "orders_90d", "orders_365"]].corr("spearman")
print(f"\nRank correlation of order counts: 30 with 90 days"
      f" {r.iloc[0, 1]:.2f},\n  90 with 365 days {r.iloc[1, 2]:.2f},"
      f" 30 with 365 days {r.iloc[0, 2]:.2f}")

# The window that touches the mark: orders dated on the mark itself,
# placed after the morning the list is made.
src = load_sources()
ctx = Context(train, src)
o = train[["contract_id", "account_id", "moment"]].merge(
    src.orders, on="account_id")
on_mark = o[o.day == o.moment].groupby("contract_id").size()
hit = train.contract_id.isin(on_mark.index)
print(f"\nContracts with an order on the mark itself: {hit.sum():,}"
      f" of {len(train):,} ({hit.mean():.1%})")
print(f"  left: {y[hit].mean():.1%} of them,"
      f" {y[~hit].mean():.1%} of the rest")
leaky = ctx.window("orders", 30).groupby("contract_id").size()
leaky = leaky.add(on_mark, fill_value=0)
leaky = leaky.reindex(train.contract_id, fill_value=0).to_numpy()
print(f"  orders_30d, AUC: {1 - auc(y, train.orders_30d):.4f} as built,"
      f" {1 - auc(y, pd.Series(leaky)):.4f} counting the mark")
