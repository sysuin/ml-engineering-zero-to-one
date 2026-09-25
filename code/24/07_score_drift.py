# What v0.6 said, list by list, and what it was worth: volume and the
# drift of its chances on the morning of each mark, then its
# calibration as each month's outcomes arrived.
import json

import pandas as pd

from foresight.evaluate import HISTORY_FROM
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.report import v06
from foresight.monitor.watch import (arrived, score_drift, scored,
                                     slope, volume)

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
model = v06().fit(train)
s = scored(model, rows[rows.moment >= GO_LIVE])
n = volume(rows).loc[s.moment.unique()]
d = score_drift(model.predict_proba(train), s)
print(f"On each mark: v0.6's chances against those it gave its"
      f" {len(train):,}\ntraining rows (average"
      f" {model.predict_proba(train).mean():.1%})")
print(f"{'list':<12}{'contracts':>10}{'usual':>7}{'average':>9}"
      f"{'PSI':>6}")
for mark, r in d.iterrows():
    print(f"{mark:%Y-%m-%d}{int(r.contracts):>10}"
          f"{n.usual[mark]:>7.0f}{r['mean']:>9.1%}{r.psi:>6.2f}")

print("\nAs outcomes arrived: every 2025 outcome on record by the day")
print(f"{'day':<12}{'lists':>6}{'contracts':>10}{'expected':>10}"
      f"{'left':>6}{'ratio':>7}{'slope':>7}")
days = pd.date_range("2025-05-01", "2026-01-01", freq="MS")
for day in days:
    a = arrived(s, day)
    sl, _ = slope(s, day)
    print(f"{day:%Y-%m-%d}{len(a):>6}{a.contracts.sum():>10,}"
          f"{a.expected.sum():>10.1f}{a.left.sum():>6}"
          f"{a.left.sum() / a.expected.sum():>7.2f}{sl:>7.2f}")
k = known(s, days[-1])
print(f"\nAll of it: v0.6's chances averaged {k.chance.mean():.1%};"
      f" {k.not_renewed.mean():.1%} left")
with open("code/24/07_score_drift.json", "w") as f:
    json.dump({f"{m:%Y-%m-%d}": round(float(v), 4)
               for m, v in d.psi.items()}, f, indent=1)
