# Training-serving skew: the twelve lists of 2024, each scored twice by
# the same fitted model, once from the training table's rows and once
# from rows whose order and ticket columns come from the scorer's SQL.
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, hits_at_k, known_by
from foresight.pipeline.model import build

LIVE = Path(__file__).with_name("live.sql").read_text()
SQL_COLUMNS = ["days_since_order", "orders_90d", "orders_prev_90d",
               "spend_365", "tickets_90d"]
table = pd.read_parquet(TABLE)
history = table[table.end_date >= HISTORY_FROM]
con = sqlite3.connect(ML_WAREHOUSE)


def served(cohort, mark):
    """The cohort's rows, with the SQL's columns for the table's."""
    live = pd.read_sql_query(LIVE, con, params={"mark": f"{mark:%F}"})
    live["days_since_order"] = live.days_since_order.astype("Int64")
    out = cohort.drop(columns=SQL_COLUMNS).merge(live, on="contract_id")
    return out[cohort.columns]


def top(p, ids, k=40):
    return set(ids[np.lexsort((ids, -p))[:k]])


print(f"{'mark':<11}{'rows':>5}{'rows that':>11}{'largest':>9}"
      f"{'calls':>7}{'leavers':>12}")
print(f"{'':<16}{'differ':>11}{'change':>9}{'moved':>7}"
      f"{'table':>7}{'SQL':>5}")
cohorts, worst = [], {"change": 0}
made_in_2024 = table.moment.between("2024-01-01", "2024-12-30")
for mark, cohort in table[made_in_2024].groupby("moment"):
    live = served(cohort, mark)
    rows = known_by(history, mark)
    model = build().fit(rows, rows.not_renewed)
    p, q = (model.predict_proba(r)[:, 1] for r in (cohort, live))
    a, b = (r[SQL_COLUMNS].astype(float).to_numpy()
            for r in (cohort, live))
    differ = int((np.abs(a - b) > 0.005).any(axis=1).sum())
    ids, y = cohort.contract_id.to_numpy(), cohort.not_renewed
    moved = len(top(p, ids) - top(q, ids))
    hits = [hits_at_k(y, s, ids) for s in (p, q)]
    print(f"{mark:%Y-%m-%d} {len(cohort):>5}{differ:>11}"
          f"{np.abs(p - q).max():>9.3f}{moved:>7}{hits[0]:>7}"
          f"{hits[1]:>5}")
    cohorts.append({"mark": f"{mark:%F}", "rows": len(cohort),
                    "differ": differ, "moved": moved, "hits": hits})
    i = int(np.argmax(np.abs(p - q)))
    if abs(p[i] - q[i]) > worst["change"]:
        worst = {"change": abs(p[i] - q[i]), "mark": f"{mark:%F}",
                 "contract": int(ids[i]),
                 "account": int(cohort.account_id.iloc[i]),
                 "table": {c: float(a[i, j]) for j, c in
                           enumerate(SQL_COLUMNS)},
                 "sql": {c: float(b[i, j]) for j, c in
                         enumerate(SQL_COLUMNS)},
                 "chance": [float(p[i]), float(q[i])],
                 "rank": [int((p > p[i]).sum()) + 1,
                          int((q > q[i]).sum()) + 1]}
total = [sum(c[k] for c in cohorts) for k in ("differ", "moved")]
print(f"{'2024':<16}{total[0]:>11}{'':>9}{total[1]:>7}"
      f"{sum(c['hits'][0] for c in cohorts):>7}"
      f"{sum(c['hits'][1] for c in cohorts):>5}")

w = worst
print(f"\nContract {w['contract']}, account {w['account']}, marked"
      f" {w['mark']}")
print(f"{'':<18}{'table':>9}{'SQL':>9}")
for c in SQL_COLUMNS:
    a, b = (f"{v:,.0f}" if v == v else "none"     # NaN: no order
            for v in (w["table"][c], w["sql"][c]))
    print(f"{c:<18}{a:>9}{b:>9}")
print(f"{'chance':<18}{w['chance'][0]:>9.1%}{w['chance'][1]:>9.1%}")
print(f"{'place in cohort':<18}{w['rank'][0]:>9}{w['rank'][1]:>9}")
with open("code/21/03_skew.json", "w") as f:
    json.dump({"cohorts": cohorts, "worst": w}, f)
