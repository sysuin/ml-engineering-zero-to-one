# The account manager two ways: today's, from accounts, and the one in
# force on the day before the mark, from account_history.
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso

DESK = "Retention desk"
con = sqlite3.connect(ML_WAREHOUSE)
now = pd.read_sql_query(
    "SELECT account_id, account_manager AS now FROM accounts", con)
history = pd.read_sql_query("""SELECT account_id, valid_from,
    account_manager AS at_mark FROM account_history""", con)
history["valid_from"] = pd.to_datetime(history.valid_from)
table = pd.read_parquet(TABLE).merge(now, on="account_id")
asof = pd.merge_asof(table.sort_values("moment"),
                     history.sort_values("valid_from"),
                     left_on="moment", right_on="valid_from",
                     by="account_id", allow_exact_matches=False)
table = table.merge(asof[["contract_id", "at_mark"]], on="contract_id")
table["at_mark"] = table.at_mark.fillna("(no history yet)")
table["desk_now"] = (table.now == DESK) * 1.0
table["desk_then"] = (table.at_mark == DESK) * 1.0

train = table[table.end_date.between("2023-01-01", "2024-06-30")]
print("On the Retention desk        today   at the mark")
for name, y in (("leavers", 1), ("renewers", 0)):
    rows = train[train.not_renewed == y]
    print(f"  training {name:<17}{rows.desk_now.mean():>8.1%}"
          f"{rows.desk_then.mean():>13.1%}")
later = table.groupby("account_id").not_renewed.max()
renew = train[(train.not_renewed == 0) & (train.desk_now == 1)]
print(f"  renewers on the desk today: {len(renew):,}, and"
      f" {later[renew.account_id].mean():.1%} of their\n  accounts"
      " left at a later renewal")

# One leaver's history, beside the dates that matter.
valid = table[table.end_date.between(*SPLITS["validation"])]
leaver = valid[(valid.not_renewed == 1) & (valid.desk_now == 1)].iloc[0]
rows = history[history.account_id == leaver.account_id]
print(f"\nAccount {leaver.account_id}: mark {leaver.moment:%Y-%m-%d},"
      f" contract ends {leaver.end_date:%Y-%m-%d}")
for _, h in rows.iterrows():
    print(f"  from {h.valid_from:%Y-%m-%d}  {h.at_mark}")


def score(extra, column_at_mark=None):
    """The lasso with `extra`, backtested on validation; each cohort
    scored with `extra` replaced by `column_at_mark` if given."""
    hist = table[table.end_date >= HISTORY_FROM]
    out = []
    for mark, c in valid.groupby("moment"):
        model = FeaturedLasso(extra).fit(known_by(hist, mark))
        if column_at_mark:
            c = c.assign(**{extra[0]: c[column_at_mark]})
        out.append(c.assign(model=model.predict_proba(c)))
    return pd.concat(out)


print("\nValidation, 240 calls                leavers     AUC")
out = {}
for name, s in (("v0.4's lasso", score(())),
                ("with today's manager", score(("desk_now",))),
                ("the same, on the morning",
                 score(("desk_now",), "desk_then"))):
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    by = [auc(c.not_renewed, c.model) for _, c in s.groupby("moment")]
    out[name] = {"hits": hits, "auc": auc(s.not_renewed, s.model),
                 "cohorts": by}
    print(f"  {name:<34}{hits:>7}{out[name]['auc']:>8.3f}")

# Chapter 11's honest version: the share of a manager's contracts that
# left in the year before the mark, managers as they were at the time.
ended = table[["end_date", "at_mark", "not_renewed"]]
rates = {}
for (who, mark), _ in train.groupby(["at_mark", "moment"]):
    past = ended[(ended.at_mark == who) & (ended.end_date < mark)
                 & (ended.end_date >= mark - pd.Timedelta(days=365))]
    rates[who, mark] = (past.not_renewed.sum() + 20 * 0.065) / (
        len(past) + 20)
rate = [rates[k] for k in zip(train.at_mark, train.moment)]
a = auc(train.not_renewed, rate)
print(f"\nManager's past-year leaving rate, alone: AUC"
      f" {max(a, 1 - a):.3f} on training")
out["example"] = {"account": int(leaver.account_id),
                  "moment": str(leaver.moment.date()),
                  "end": str(leaver.end_date.date()),
                  "history": [[str(h.valid_from.date()), h.at_mark]
                              for _, h in rows.iterrows()]}
with open("code/13/02_process_leak.json", "w") as f:
    json.dump(out, f)
