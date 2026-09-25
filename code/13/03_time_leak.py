# Two columns that read past the mark: the notice date, and windows
# that end when the contract ends instead of at the mark.
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, events, legacy_ids
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso

con = sqlite3.connect(ML_WAREHOUSE)
notice = pd.read_sql_query(
    "SELECT contract_id, notice_date FROM contracts", con)
table = pd.read_parquet(TABLE).merge(notice, on="contract_id")
table["notice_date"] = pd.to_datetime(table.notice_date)
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
given = train[train.notice_date.notna()]
before = (given.end_date - given.notice_date).dt.days
print(f"notice_date: filled for {len(given):,} training contracts,"
      f" {given.not_renewed.mean():.0%} of\n  them leavers;"
      f" {before.min()} to {before.max()} days before the end;"
      f" on record by\n  the mark for"
      f" {(given.notice_date < given.moment).sum()} of them")

# Windows anchored on the end date instead of the mark.
orders, tickets = events(con, legacy_ids(con))
keys = table[["contract_id", "account_id", "end_date"]]
t = keys.merge(tickets, on="account_id")
age = (t.end_date - t.day).dt.days
t = t[(age >= 1) & (age <= 90)]
table["tickets_to_end"] = table.contract_id.map(
    t.groupby("contract_id").size()).fillna(0)
o = keys.merge(orders, on="account_id")
o = o[o.day < o.end_date]
last = o.groupby("contract_id").day.max()
table["gap_to_end"] = (table.end_date - table.contract_id.map(last)
                       ).dt.days.fillna(365).astype(float)

valid = table[table.end_date.between(*SPLITS["validation"])]
print("\nValidation            leavers  renewers   alone, AUC")
for col in ("tickets_90d", "tickets_to_end",
            "days_since_order", "gap_to_end"):
    x = valid[col].astype(float).fillna(365)
    m = x.groupby(valid.not_renewed).mean()
    a = auc(valid.not_renewed, x)
    print(f"  {col:<20}{m[1]:>8.2f}{m[0]:>10.2f}{max(a, 1 - a):>13.3f}")


def score(extra):
    hist = table[table.end_date >= HISTORY_FROM]
    out = []
    for mark, c in valid.groupby("moment"):
        model = FeaturedLasso(extra).fit(known_by(hist, mark))
        out.append(c.assign(model=model.predict_proba(c)))
    return pd.concat(out)


print("\nValidation, 240 calls                leavers     AUC")
out = {}
for name, extra in (("v0.4's lasso", ()),
                    ("with tickets to the end", ("tickets_to_end",)),
                    ("with the gap to the end", ("gap_to_end",))):
    s = score(extra)
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    out[name] = {"hits": hits, "auc": auc(s.not_renewed, s.model)}
    print(f"  {name:<34}{hits:>7}{out[name]['auc']:>8.3f}")
with open("code/13/03_time_leak.json", "w") as f:
    json.dump(out, f)
