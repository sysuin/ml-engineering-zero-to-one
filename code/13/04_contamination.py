# A column computed across the whole table, validation and test rows
# included, against the same idea computed as of each row's mark.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso

table = pd.read_parquet(TABLE)
# "How many renewals has this account had?", two ways.
table["account_rows"] = table.groupby(
    "account_id").contract_id.transform("size").astype(float)
pairs = table[["contract_id", "account_id", "moment"]].merge(
    table[["account_id", "end_date"]], on="account_id")
ended = pairs[pairs.end_date < pairs.moment]
table["rows_before"] = table.contract_id.map(
    ended.groupby("contract_id").size()).fillna(0.0)

valid = table[table.end_date.between(*SPLITS["validation"])]
print("Validation          leavers  renewers   alone, AUC")
for col in ("account_rows", "rows_before"):
    m = valid.groupby("not_renewed")[col].mean()
    a = auc(valid.not_renewed, valid[col])
    print(f"  {col:<18}{m[1]:>8.2f}{m[0]:>10.2f}"
          f"{max(a, 1 - a):>13.3f}")

hist = table[table.end_date >= HISTORY_FROM]
print("\nValidation, 240 calls                leavers     AUC")
out = {}
for name, extra in (("v0.4's lasso", ()),
                    ("with account_rows", ("account_rows",)),
                    ("with rows_before", ("rows_before",))):
    s = pd.concat(c.assign(model=FeaturedLasso(extra).fit(
                      known_by(hist, mark)).predict_proba(c))
                  for mark, c in valid.groupby("moment"))
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    out[name] = {"hits": hits, "auc": auc(s.not_renewed, s.model)}
    print(f"  {name:<34}{hits:>7}{out[name]['auc']:>8.3f}")

# Where account_rows gets its knowledge: rows after the split.
test = table[table.end_date >= SPLITS["test"][0]]
seen = valid.account_id.isin(test.account_id)
left = seen[valid.not_renewed == 1].mean()
print(f"\n{seen.mean():.1%} of validation contracts have an account-"
      f"mate in the test\nyear; among leavers, {left:.1%}")
with open("code/13/04_contamination.json", "w") as f:
    json.dump(out, f)
