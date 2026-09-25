# A label defined from orders, "no order in the six months before the
# contract ends", beside Foresight's label, and the rule on both.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, events, legacy_ids
from foresight.evaluate import SPLITS, auc
from foresight.models.logistic import days_since_rule

con = sqlite3.connect(ML_WAREHOUSE)
orders, _ = events(con, legacy_ids(con))
table = pd.read_parquet(TABLE)
pairs = table[["contract_id", "account_id", "end_date"]].merge(
    orders, on="account_id")
age = (pairs.end_date - pairs.day).dt.days
busy = pairs[(age >= 1) & (age <= 180)].contract_id.unique()
table["silent"] = (~table.contract_id.isin(busy)) * 1

valid = table[table.end_date.between(*SPLITS["validation"])]
print("Validation contracts        not renewed   renewed")
for name, s in (("silent for six months", 1), ("ordered", 0)):
    rows = valid[valid.silent == s]
    print(f"  {name:<26}{rows.not_renewed.sum():>11,}"
          f"{(1 - rows.not_renewed).sum():>10,}")

# Half the silent label's window lies before the mark: any order in
# the 90 days before it settles the label as 0 that morning.
settled = valid.orders_90d > 0
print(f"\nSettled on the morning of the mark: {settled.mean():.1%} of"
      f" contracts,\n  every one of them labelled"
      f" {valid[settled].silent.max()}")
print("\nThe days-since rule, ranking      AUC")
for name, label in (("not renewed (Foresight)", "not_renewed"),
                    ("silent for six months", "silent")):
    a = auc(valid[label], days_since_rule(valid))
    print(f"  {name:<30}{a:.3f}")
