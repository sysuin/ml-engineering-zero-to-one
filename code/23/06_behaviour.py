# Behavioural tests: what the model does when one input moves and the
# rest stay still. First the lasso's weights on the inputs the tests
# move, then the tests themselves, then one contract before and after
# each change, and last what the training rows say about tickets.
import json

import pandas as pd
from _suite import run

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
print("the lasso's weights (standardised inputs):")
for c in ("days_since_order", "discount_pct", "tickets_90d"):
    print(f"  {c:<18}{w[c]:+.3f}")

r = run("tests/test_behaviour.py")
print(f"\n{'behavioural test':<52}{'result':>14}")
for name, g in r.groupby("name", sort=False):
    out = g.outcome.value_counts()
    said = ", ".join(f"{n} {o}" for o, n in out.items())
    print(f"{name.removeprefix('test_')[:52]:<52}{said:>14}")

latest = rows[rows.moment == rows.moment.max()]
cohort = latest[latest.region == "Midwest"]
p = pd.Series(fitted.predict_proba(cohort)[:, 1],
              index=cohort.index)
one = cohort.loc[(p - p.median()).abs().sort_values(
    kind="stable").index[:1]]              # the median Midwest chance


def chance(row):
    try:
        return float(fitted.predict_proba(row)[0, 1])
    except ValueError:
        return None


pairs = [("name changed", one.assign(name="Hartton Hygiene Ltd")),
         ("region spelt 'Mid-west'", one.assign(region="Mid-west")),
         ("gap 90 days longer", one.assign(
             days_since_order=one.days_since_order + 90)),
         ("discount 5 points larger", one.assign(
             discount_pct=one.discount_pct.fillna(0) + 5)),
         ("3 more tickets", one.assign(
             tickets_90d=one.tickets_90d + 3))]
before = chance(one)
c = one.iloc[0]
print(f"\ncontract {c.contract_id}, marked {c.moment:%Y-%m-%d}:"
      f" {c.segment}, {c.region},")
print(f"  gap {c.days_since_order} days, discount {c.discount_pct},"
      f" {c.tickets_90d} ticket(s); chance {before:.2%}")
record = []
for what, row in pairs:
    after = chance(row)
    shown = "refused" if after is None else f"{after:.2%}"
    print(f"  {what:<26}{before:>8.2%} -> {shown}")
    record.append({"change": what, "before": before, "after": after})

y = rows.not_renewed
band = rows.tickets_90d.clip(upper=3)
print("\nthe rows it learned from: share that left, by tickets in"
      " the quarter")
for n, g in y.groupby(band):
    label = f"{n}+" if n == 3 else f"{n}"
    print(f"  {label:<3}{g.mean():>7.1%} of {len(g):>5,}")

with open("code/23/06_behaviour.json", "w") as f:
    json.dump({"contract": int(c.contract_id), "pairs": record}, f,
              indent=1)
