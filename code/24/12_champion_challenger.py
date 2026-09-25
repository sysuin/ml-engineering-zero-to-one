# Champion and challengers through 2025. v0.6 serves, frozen; three
# challengers refit monthly in shadow; each morning the gate reads the
# shadow record's outcomes so far. The truth file is read only at the
# end, for the ceiling.
# timeout: 400
import json

import pandas as pd

from foresight.config import TRUTH
from foresight.evaluate import HISTORY_FROM, hits_at_k
from foresight.models.regularised import compare
from foresight.monitor.cohorts import as_scored, known
from foresight.monitor.policy import (forward, paired, production,
                                      record, shadow)
from foresight.monitor.report import CONTENDERS

rows = as_scored(HISTORY_FROM, "2026-03-31")
s = shadow(rows, CONTENDERS)
names = [c.name for c in CONTENDERS]
log, served = forward(s, names[0], names[1:])

print("The gate, exposure against the champion of the day")
print(f"{'mark':<7}{'lists':>5}{'points (95%)':>20}"
      f"{'AUC within (95%)':>25}")
for g in log[(log.challenger == "exposure")
             & (log.cohorts >= 3)].itertuples():
    p, a = g.precision, g.auc
    verdict = " promoted" if g.promote else ""
    print(f"{g.day:%m-%d}  {g.cohorts:>5}{p[0] * 100:>+6.1f}"
          f" ({p[1] * 100:+.1f} to {p[2] * 100:+.1f})"
          f"{a[0]:>+7.3f} ({a[1]:+.3f} to {a[2]:+.3f}){verdict}")
promoted = served[served != names[0]]
print(f"Serving from {promoted.index[0]:%d %B}: {promoted.iloc[0]}."
      " That morning, against v0.6:")
day = promoted.index[0]
for n in ("monthly", "voss share"):
    g = log[(log.challenger == n) & (log.day == day)].iloc[0]
    print(f"  {n:<12}AUC within {g.auc[0]:+.3f}"
          f" ({g.auc[1]:+.3f} to {g.auc[2]:+.3f})")
pooled = compare(known(record(s, "exposure"), day),
                 known(record(s, "v0.6"), day))["auc"]
print(f"  exposure    AUC pooled {pooled[0]:+.3f}"
      f" ({pooled[1]:+.3f} to {pooled[2]:+.3f})")

truth = pd.read_csv(TRUTH / "renewals.csv",
                    usecols=["contract_id", "p_leave"])
done = known(production(s, served), "2026-01-01")
lists = {"as served": done} | {n: known(record(s, n), "2026-01-01")
                               for n in names}
lists["rule"] = done.assign(model=done.rule)
lists["ceiling"] = done.drop(columns="model").merge(
    truth.rename(columns={"p_leave": "model"}), on="contract_id")
calls = 40 * done.moment.nunique()
print(f"\nThe {done.moment.nunique()} lists of 2025 with outcomes,"
      f" {calls} calls, {int(done.not_renewed.sum())} leavers")
print(f"{'':<12}{'leavers':>8}{'precision':>11}"
      f"{'minus rule, points (95%)':>29}")
out = {}
for n, r in lists.items():
    hits = sum(hits_at_k(g.not_renewed, g.model, g.contract_id)
               for _, g in r.groupby("moment"))
    out[n] = hits
    line = f"{n:<12}{hits:>8}{hits / calls:>11.1%}"
    if n not in ("rule", "ceiling"):
        p = paired(r, lists["rule"])["precision"]
        line += (f"{p[0] * 100:>+13.1f} ({p[1] * 100:+.1f} to"
                 f" {p[2] * 100:+.1f})")
    print(line)
with open("code/24/12_champion_challenger.json", "w") as f:
    json.dump({"hits": out, "calls": calls,
               "served": {f"{k:%Y-%m-%d}": v
                          for k, v in served.items()}}, f, indent=1)
