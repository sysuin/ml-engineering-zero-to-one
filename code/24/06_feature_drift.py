# PSI for every column v0.6 reads, list by list through 2025, against
# the rows it learned from, with bins fixed on those rows.
import json

from foresight.evaluate import HISTORY_FROM
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.psi import ACT, Baseline
from foresight.train import COLUMNS

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
base = Baseline(train, COLUMNS + ["supplier_voss"])
table = base.table(rows[rows.moment >= "2024-07-01"])

SHORT = ["gap", "o90", "o180", "spnd", "tix", "ten", "seg", "reg",
         "term", "leg", "disc", "voss"]
print("list " + "".join(f"{s:>5}" for s in SHORT))
for mark, r in table[table.index >= GO_LIVE].iterrows():
    print(f"{mark:%m-%d}" + "".join(f"{v:>5.2f}" for v in r))
year = table[table.index >= GO_LIVE]
print("mean " + "".join(f"{v:>5.2f}" for v in year.mean()))
print(f"\nLists at or over {ACT} in 2025, by column:")
for c, n in (year >= ACT).sum().items():
    if n:
        print(f"  {c:<16}{n:>3} of {len(year)}")
with open("code/24/06_feature_drift.json", "w") as f:
    json.dump({c: {f"{m:%Y-%m-%d}": round(v, 4)
                   for m, v in table[c].items()}
               for c in table.columns}, f, indent=1)
