# Voss's share of each long-tail account's spend, and what it meant,
# in three stretches of marks: did the inputs move, or the outcome?
import json

import pandas as pd

from foresight.monitor.cohorts import as_scored, known
from foresight.monitor.psi import Baseline

rows = known(as_scored("2023-01-01", "2025-12-31"), "2026-01-01")
rows = rows[rows.key_account == 0]
PERIODS = {"2023": ("2023-01-01", "2023-12-31"),
           "Aug24-Jan25": ("2024-08-01", "2025-01-31"),
           "Mar-Oct25": ("2025-02-01", "2025-10-31")}
BANDS = [-1, 0, 0.1, 0.2, 0.3, 1]
NAMES = ["none", "up to 10%", "10 to 20%", "20 to 30%", "over 30%"]
part = {k: rows[rows.moment.between(*v)] for k, v in PERIODS.items()}

print("Voss's share of the year's spend, by the contract's mark")
print(f"{'':18}" + "".join(f"{k:>14}" for k in part))
print(f"{'contracts':<18}" + "".join(f"{len(p):>14,}"
                                     for p in part.values()))
print(f"{'average share':<18}" + "".join(
    f"{p.supplier_voss.mean():>14.1%}" for p in part.values()))
print(f"{'over 30%':<18}" + "".join(
    f"{(p.supplier_voss > 0.3).mean():>14.1%}" for p in part.values()))
names = list(part)
psi = [Baseline(part[a], ["supplier_voss"]).column(part[b],
                                                  "supplier_voss")
       for a, b in zip(names, names[1:])]
print(f"{'PSI, period before':<18}{'':>14}" + "".join(
    f"{v:>14.2f}" for v in psi))

print("\nShare that did not renew, by Voss's share of spend")
out = {"periods": names, "bands": NAMES, "rate": {}, "mix": {}}
for k, p in part.items():
    band = pd.cut(p.supplier_voss, BANDS, labels=NAMES)
    out["rate"][k] = p.groupby(band, observed=False).not_renewed.mean(
        ).round(4).tolist()
    out["mix"][k] = band.value_counts(normalize=True).reindex(
        NAMES).round(4).tolist()
for i, name in enumerate(NAMES):
    print(f"{name:<18}" + "".join(f"{out['rate'][k][i]:>14.1%}"
                                  for k in names))
print(f"{'all':<18}" + "".join(f"{p.not_renewed.mean():>14.1%}"
                               for p in part.values()))
with open("code/24/01_two_drifts.json", "w") as f:
    json.dump(out, f, indent=1)
