# Exercise 1: a table with three planted leaks, and the checks that find
# them. planted() is the exercise; read the rest after trying it.
import sqlite3

import pandas as pd

from foresight.checks.leakage import adversarial, screen
from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, events, legacy_ids
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso


def planted() -> pd.DataFrame:
    """Chapter 4's table plus three columns: col_a, col_b, col_c."""
    con = sqlite3.connect(ML_WAREHOUSE)
    t = pd.read_parquet(TABLE)
    _, tickets = events(con, legacy_ids(con))
    k = t[["contract_id", "account_id", "end_date"]].merge(
        tickets, on="account_id")
    last = k[k.day < k.end_date].groupby("contract_id").day.max()
    t["col_a"] = (t.end_date - t.contract_id.map(last)).dt.days
    t["col_a"] = t.col_a.fillna(999).astype(float)
    closed = pd.read_sql_query(
        "SELECT account_id, closed_on FROM accounts", con)
    t = t.merge(closed, on="account_id")
    t["col_b"] = t.closed_on.notna().map({True: "closed",
                                          False: "open"})
    t["col_c"] = t.groupby("account_id").not_renewed.transform("mean")
    return t.drop(columns="closed_on")


table = planted()
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
valid = table[table.end_date.between(*SPLITS["validation"])]
cols = ["col_a", "col_b", "col_c"]
print("The screen, training rows")
for c, r in screen(train, cols).iterrows():
    print(f"  {c}  AUC {r.auc:.3f}  {r.screen}")

hist = table[table.end_date >= HISTORY_FROM]
print("\nThe lasso with each, validation     leavers     AUC")
for extra in ((), ("col_a",), ("col_c",)):
    s = pd.concat(c.assign(model=FeaturedLasso(extra).fit(
                      known_by(hist, m)).predict_proba(c))
                  for m, c in valid.groupby("moment"))
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    name = extra[0] if extra else "none"
    print(f"  {name:<32}{hits:>9}{auc(s.not_renewed, s.model):>8.3f}")

together, each = adversarial(train, valid, cols)
print("\nTraining against validation, each column alone")
for c, r in each.iterrows():
    print(f"  {c}  AUC {r['shift auc']:.3f}  {r['shift']}")
