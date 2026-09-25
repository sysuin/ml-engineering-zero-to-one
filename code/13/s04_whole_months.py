# Exercise 4: tenure in whole months instead of days, against the
# calendar adversarial validation found in tenure_days.
import pandas as pd

from foresight.checks.leakage import adversarial
from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, SPLITS, auc, hits_at_k,
                                known_by)
from foresight.models.featured import FeaturedLasso

table = pd.read_parquet(TABLE)
since = table.moment - pd.to_timedelta(table.tenure_days, unit="D")
table["tenure_months"] = ((table.moment.dt.year - since.dt.year) * 12
                          + table.moment.dt.month - since.dt.month
                          - (table.moment.dt.day < since.dt.day))
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
valid = table[table.end_date.between(*SPLITS["validation"])]
_, each = adversarial(train, valid, ["tenure_days", "tenure_months"])
print("Telling training from validation, alone")
for c, r in each.iterrows():
    print(f"  {c:<15}AUC {r['shift auc']:.3f}  {r['shift']}")

hist = table[table.end_date >= HISTORY_FROM]
print("\nThe lasso on validation              leavers     AUC")
for name, rows in (("tenure in days", table),
                   ("tenure in whole months",
                    table.assign(tenure_days=table.tenure_months))):
    h = rows[rows.end_date >= HISTORY_FROM]
    s = pd.concat(c.assign(model=FeaturedLasso().fit(
                      known_by(h, m)).predict_proba(c))
                  for m, c in rows[rows.end_date.between(
                      *SPLITS["validation"])].groupby("moment"))
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    print(f"  {name:<34}{hits:>7}{auc(s.not_renewed, s.model):>8.3f}")
