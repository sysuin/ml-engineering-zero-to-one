# Every column alone against the ceiling: Chapter 4's columns, the
# library's 36, and the leaky candidates from this chapter, screened on
# the training contracts. The ceiling of these rows is read from the
# generator for one line of comparison, as in 01_target_leak.
import json
import sqlite3

import pandas as pd

from foresight.checks.leakage import CEILING_AUC, REVIEW_SHARE, screen
from foresight.config import ML_WAREHOUSE, TRUTH
from foresight.data.build_table import events, legacy_ids
from foresight.evaluate import auc
from foresight.features.build import load
from foresight.features.registry import names
from foresight.train import COLUMNS

con = sqlite3.connect(ML_WAREHOUSE)
table = load().merge(pd.read_sql_query("""SELECT contract_id,
    cancellation_reason, notice_date FROM contracts""", con),
    on="contract_id").merge(pd.read_sql_query("""SELECT account_id,
    account_manager AS manager_today FROM accounts""", con),
    on="account_id")
table["notice_date"] = pd.to_datetime(table.notice_date)
table["account_rows"] = table.groupby(
    "account_id").contract_id.transform("size")
_, tickets = events(con, legacy_ids(con))
t = table[["contract_id", "account_id", "end_date"]].merge(
    tickets, on="account_id")
age = (t.end_date - t.day).dt.days
table["tickets_to_end"] = table.contract_id.map(
    t[(age >= 1) & (age <= 90)].groupby("contract_id").size()).fillna(0)

LEAKY = ["cancellation_reason", "notice_date", "manager_today",
         "account_rows", "tickets_to_end"]
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
result = screen(train, COLUMNS + names() + LEAKY)
result = result.sort_values("auc", ascending=False)

truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
best = auc(train.not_renewed, truth.p_leave.reindex(train.contract_id))
review = 0.5 + REVIEW_SHARE * (CEILING_AUC - 0.5)
print(f"{len(result)} columns, {len(train):,} training contracts."
      f" Fail at {CEILING_AUC},\nreview from {review:.3f}; the true"
      f" chances on these rows score {best:.3f}\n")
print(f"{'column':<22}{'alone, AUC':>11}  verdict")
for c, r in result.head(12).iterrows():
    print(f"{c:<22}{r.auc:>11.3f}  {r.screen}")
print(f"... and {len(result) - 12} more, from"
      f" {result.auc.iloc[12]:.3f} down to {result.auc.min():.3f}")
print(f"\n{'':22}{'fail':>6}{'review':>8}{'pass':>6}")
for name, cols in (("Chapter 4's columns", COLUMNS),
                   ("the library", names()),
                   ("this chapter's leaks", LEAKY)):
    v = result.loc[cols, "screen"].value_counts()
    print(f"{name:<22}{v.get('fail', 0):>6}{v.get('review', 0):>8}"
          f"{v.get('pass', 0):>6}")
with open("code/13/08_single_feature_screen.json", "w") as f:
    json.dump({"ceiling": CEILING_AUC, "review": review, "best": best,
               "leaky": LEAKY, "auc": result.auc.round(4).to_dict()},
              f)
