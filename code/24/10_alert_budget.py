# 2025 replayed through the monitor, following v0.6's lists all year:
# every alert a naive monitor would raise, then the alerts left once
# drift must move the score and a day's feed alerts are one alert.
from collections import Counter

from foresight.evaluate import HISTORY_FROM
from foresight.monitor.alerts import year
from foresight.monitor.cohorts import (GO_LIVE, RECORD_ENDS, as_scored,
                                       known)
from foresight.monitor.feeds import alerts as feed_alerts
from foresight.monitor.feeds import daily
from foresight.monitor.report import v06
from foresight.monitor.watch import scored
from foresight.train import COLUMNS

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
model = v06().fit(train)
s = scored(model, rows[rows.moment >= GO_LIVE])
feed = feed_alerts(daily())
watched = COLUMNS + ["supplier_voss"]
args = ((model, train), s, watched, GO_LIVE, RECORD_ENDS, feed)
naive = year(*args, budget=False, due=rows)
kept = year(*args, due=rows)

print(f"{'check':<10}{'naive':>7}{'budget':>8}")
a, b = Counter(x.check for x in naive), Counter(x.check for x in kept)
for check in sorted(a):
    print(f"{check:<10}{a[check]:>7}{b[check]:>8}")
print(f"{'all':<10}{len(naive):>7}{len(kept):>8}"
      f"   ({len(naive) / 12:.1f} and {len(kept) / 12:.1f} a month)")
inputs = Counter(x.subject for x in naive if x.check == "inputs")
print("Naive input alerts:")
for k, v in inputs.items():
    print(f"  {k:<18}{v:>3}")
print("\nThe alerts that survive the budget")
for x in kept:
    print(x.line())
