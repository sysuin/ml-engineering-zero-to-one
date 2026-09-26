# Exercise 1: "more orders last quarter never raise the risk", as a
# directional test on v0.6, with the weights that decide it and what
# happens when both quarters move together.
import pandas as pd

from foresight import gate
from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.pipeline.model import weights

table = pd.read_parquet(TABLE)
model, cfg = gate.candidate()
rows = known_by(table[table.end_date >= HISTORY_FROM],
                cfg["data"]["as_of"])
fitted = model.fit(rows, rows.not_renewed)
w = weights(fitted)
for c in ("orders_90d", "orders_prev_90d"):
    print(f"{c:<16}{w[c]:+.3f}  (standard deviation"
          f" {rows[c].std():.2f} orders)")

latest = sorted(rows.moment.unique())[-6:]
cohorts = rows[rows.moment.isin(latest)]


def chance(r):
    return fitted.predict_proba(r)[:, 1]


before = chance(cohorts)
last = chance(cohorts.assign(orders_prev_90d=cohorts.orders_prev_90d
                             + 3))
both = chance(cohorts.assign(orders_prev_90d=cohorts.orders_prev_90d
                             + 3, orders_90d=cohorts.orders_90d + 3))
print(f"\n{len(cohorts):,} contracts, 3 more orders:")
for what, after in (("last quarter only", last),
                    ("both quarters", both)):
    move = (after - before) * 100
    print(f"  {what:<18} risk rose for {(after > before).mean():.0%};"
          f" by {move.mean():.3f} points on average")
