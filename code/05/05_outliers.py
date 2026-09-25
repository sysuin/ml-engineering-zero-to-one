# The rows at the far end of spend: mistakes, or the key accounts?
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
con = sqlite3.connect(ML_WAREHOUSE)
accounts = pd.read_sql_query(
    "SELECT account_id, name, is_key_account AS key FROM accounts", con)
con.close()
train = train.merge(accounts, on="account_id")

# First, values that cannot be true of any account.
impossible = {
    "negative spend": (train.spend_365 < 0).sum(),
    "negative counts": (train[["orders_90d", "orders_prev_90d",
                               "tickets_90d"]] < 0).any(axis=1).sum(),
    "order before the record": (train.days_since_order > (
        train.moment - pd.Timestamp("2022-01-01")).dt.days).sum(),
}
for what, n in impossible.items():
    print(f"{what:26}{n:>4} rows")

q1, q3 = train.spend_365.quantile([0.25, 0.75])
flagged = train[train.spend_365 > q3 + 1.5 * (q3 - q1)]
print(f"\nBox-plot outliers in spend: {len(flagged)} rows,"
      f" {flagged.key.sum()} of them key accounts")
print(f"The five largest{'mark':>17}{'spend':>12}{'orders_90d':>11}"
      f"{'left':>5}")
for _, r in train.nlargest(5, "spend_365").iterrows():
    print(f"  {r['name']:22}{r.moment:%Y-%m-%d}{r.spend_365:>12,.0f}"
          f"{r.orders_90d:>11}{r.not_renewed:>5}")

print(f"\n{'':13}{'rows':>6}{'leavers':>9}{'rate':>7}"
      f"{'median orders_90d':>19}")
for flag, g in train.groupby("key"):
    name = "key accounts" if flag else "long tail"
    print(f"{name:13}{len(g):>6,}{g.not_renewed.sum():>9}"
          f"{g.not_renewed.mean():>7.1%}{g.orders_90d.median():>19.0f}")
tail_max = train[train.key == 0].spend_365.max()
key_spend = train[train.key == 1].spend_365
key_min = key_spend[key_spend > 0].min()
print(f"Largest long-tail spend {tail_max:,.0f}")
print(f"Smallest key-account spend above zero {key_min:,.0f}")
