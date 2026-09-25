# v0.4's lasso given one more column from the contracts table, scored
# on validation beside the ceiling, which is read from the generator's
# true chances for this comparison only.
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE, TRUTH
from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso

con = sqlite3.connect(ML_WAREHOUSE)
reason = pd.read_sql_query(
    "SELECT contract_id, cancellation_reason FROM contracts", con)
table = pd.read_parquet(TABLE).merge(reason, on="contract_id")
REASONS = sorted(table.cancellation_reason.dropna().unique())
for i, r in enumerate(REASONS):         # one column a reason
    table[f"reason_{i}"] = (table.cancellation_reason == r) * 1.0
LEAK = [f"reason_{i}" for i in range(len(REASONS))]

train = table[table.end_date.between("2023-01-01", "2024-06-30")]
filled = train.cancellation_reason.notna()
print("Training contracts   reason filled   empty")
for name, y in (("not renewed", 1), ("renewed", 0)):
    rows = train.not_renewed == y
    print(f"  {name:<19}{(rows & filled).sum():>12,}"
          f"{(rows & ~filled).sum():>8,}")


def score(extra=(), on_the_morning=False):
    """Chapter 8's backtest of the lasso with `extra` columns. With
    on_the_morning, each cohort is scored with the columns as they
    stood at its mark: no reason is on record for anyone yet."""
    history = table[table.end_date >= HISTORY_FROM]
    out = []
    for mark, c in table[table.end_date.between(
            *SPLITS["validation"])].groupby("moment"):
        model = FeaturedLasso(extra).fit(known_by(history, mark))
        if on_the_morning:
            c = c.assign(**{col: 0.0 for col in extra})
        out.append(c.assign(model=model.predict_proba(c)))
    return pd.concat(out)


def summary(s):
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    return hits, auc(s.not_renewed, s.model)


truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
clean = score()
best = clean.assign(model=truth.p_leave.reindex(
    clean.contract_id).to_numpy())
lists = {"v0.4's lasso": clean, "with the reason": score(LEAK),
         "ceiling (true chances)": best,
         "the same, on the morning": score(LEAK, on_the_morning=True)}
print("\nValidation, 6 cohorts, 240 calls     leavers     AUC")
out = {}
for name, s in lists.items():
    hits, a = summary(s)
    out[name] = {"hits": hits, "auc": a}
    print(f"  {name:<34}{hits:>7}{a:>8.3f}")
perfect = clean.groupby("moment").not_renewed.sum().clip(upper=40)
print(f"  {'a perfect list':<34}{perfect.sum():>7}")
with open("code/13/01_target_leak.json", "w") as f:
    json.dump(out, f)
