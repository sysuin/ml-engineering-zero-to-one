# Exercise 1: the first day each check said anything in 2025, and what
# a looser notice check would have bought.
from foresight.evaluate import HISTORY_FROM
from foresight.monitor import watch
from foresight.monitor.alerts import first, year
from foresight.monitor.cohorts import (GO_LIVE, RECORD_ENDS, as_scored,
                                       known)
from foresight.monitor.feeds import alerts, daily
from foresight.monitor.report import v06
from foresight.train import COLUMNS

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
model = v06().fit(train)
s = watch.scored(model, rows[rows.moment >= GO_LIVE])
found = year((model, train), s, COLUMNS, GO_LIVE, RECORD_ENDS,
             alerts(daily()), due=rows)
for check in ("feed", "volume", "inputs", "scores", "notices",
              "outcomes"):
    a = first(found, check)
    said = f"{a.day:%Y-%m-%d}  {a.subject}" if a else "never"
    print(f"{check:<10}{said}")

print("\nThe notice check at looser tail probabilities")
e = watch.early(s, "2025-12-31")
for alpha in (0.001, 0.01, 0.05):
    low, high = watch.limits(e.expected, alpha)
    hit = e[(e.noticed > high) | (e.noticed < low)]
    lists = ", ".join(f"{m:%b}" for m in hit.index)
    print(f"  alpha {alpha:<7}{len(hit)} lists: {lists}")
