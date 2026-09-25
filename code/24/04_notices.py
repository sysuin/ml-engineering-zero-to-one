# Notices as the early warning: each 2025 list's notices against the
# leavers v0.6's chances expected, then the March list taken apart on
# the morning its alert fired.
import json

import pandas as pd

from foresight.evaluate import HISTORY_FROM
from foresight.monitor.cohorts import GO_LIVE, as_scored, known, noticed
from foresight.monitor.report import v06
from foresight.monitor.watch import early, scored

rows = as_scored(HISTORY_FROM, "2026-03-31")
model = v06().fit(known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE))
s = scored(model, rows[rows.moment >= GO_LIVE])

e = early(s, "2025-12-31")
print(f"{'list':<12}{'contracts':>10}{'expected':>10}{'notices':>9}"
      f"{'limits':>12}")
for mark, r in e.iterrows():
    flag = "  alert" if r.alert else ""
    print(f"{mark:%Y-%m-%d}{r.contracts:>10}{r.expected:>10.1f}"
          f"{r.noticed:>9}{r.low:>7.0f} to{r.high:>3.0f}{flag}")

DAY = pd.Timestamp("2025-04-02")
march = s[s.moment == "2025-03-02"].copy()
march["gave notice"] = noticed(march, DAY)
march["band"] = pd.cut(march.supplier_voss, [-1, 0.1, 0.3, 1],
                       labels=["Voss under 10%", "10 to 30%",
                               "over 30%"])
print(f"\nThe list of 2 March, as it stood on {DAY.day} {DAY:%B}")
print(f"{'':16}{'contracts':>10}{'expected':>10}{'notices':>9}"
      f"{'on price':>10}")
for band, g in march.groupby("band", observed=True):
    price = g.reason.eq("Moved to a competitor on price")
    print(f"{band:<16}{len(g):>10}{g.chance.sum():>10.1f}"
          f"{int(g['gave notice'].sum()):>9}"
          f"{int((price & g['gave notice']).sum()):>10}")
with open("code/24/04_notices.json", "w") as f:
    json.dump({f"{m:%Y-%m-%d}": {k: round(float(r[k]), 2) for k in
                                 ("expected", "noticed", "low", "high")}
               for m, r in e.iterrows()}, f, indent=1)
