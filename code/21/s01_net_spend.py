# Exercise 1: a skew of your own. The serving rows are made by the
# shared definition, except that spend is net of each line's discount,
# as finance counts revenue. The skew test, then both models' lists.
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, legacy_ids
from foresight.evaluate import HISTORY_FROM, SPLITS, known_by
from foresight.models.boosting import RenewalBooster
from foresight.pipeline.features import at_marks, skew
from foresight.pipeline.model import build

table = pd.read_parquet(TABLE)
history = table[table.end_date >= HISTORY_FROM]
cohorts = table[table.end_date.between(*SPLITS["validation"])]
served = at_marks(sorted(cohorts.moment.unique()))

with sqlite3.connect(ML_WAREHOUSE) as con:
    lines = pd.read_sql_query("""
        SELECT o.account_id, o.order_date AS day,
               SUM(l.qty * l.unit_price
                   * (100 - l.discount_pct) / 100.0) AS net
        FROM orders o JOIN order_lines l USING (order_id)
        GROUP BY o.order_id""", con)
    ids = legacy_ids(con)
lines["account_id"] = lines.account_id.replace(ids)
lines["day"] = pd.to_datetime(lines.day)
keys = served[["contract_id", "account_id", "moment"]]
o = keys.merge(lines, on="account_id")
age = (o.moment - o.day).dt.days
net = o[(age >= 1) & (age <= 365)].groupby("contract_id").net.sum()
served["spend_365"] = (served.contract_id.map(net).fillna(0.0)
                       .round(2))

found = skew(cohorts, served)
print("The skew test, validation cohorts:")
for c, r in found[found.differ > 0].iterrows():
    print(f"  {c:<12}{int(r.differ):>6,} of {int(r.rows):,} rows"
          " differ")


both = cohorts.merge(served[["contract_id", "spend_365"]],
                     on="contract_id", suffixes=("", "_net"))
lower = 1 - both.spend_365_net.sum() / both.spend_365.sum()
print(f"  net spend is {lower:.1%} below list spend over these rows")


def calls(p, ids, k=40):
    return set(ids[np.lexsort((ids, -p))[:k]])


print(f"\n{'mark':<12}{'calls moved: lasso':>20}{'booster':>10}")
for mark, cohort in cohorts.groupby("moment"):
    live = (served.set_index("contract_id").loc[cohort.contract_id]
            .reset_index())
    rows = known_by(history, mark)
    lasso = build().fit(rows, rows.not_renewed)
    booster = RenewalBooster().fit(rows)
    ids_ = cohort.contract_id.to_numpy()
    moved = []
    for chance in (lambda r: lasso.predict_proba(r)[:, 1],
                   booster.predict_proba):
        a, b = calls(chance(cohort), ids_), calls(chance(live), ids_)
        moved.append(len(a - b))
    print(f"{mark:%Y-%m-%d}{moved[0]:>20}{moved[1]:>10}")
