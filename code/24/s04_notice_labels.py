# Exercise 4: treat a list's outcome as known the morning after its
# notices were due, 60 days before the end, for retraining and for the
# gate. When would the exposure challenger have been promoted?
# timeout: 400
import pandas as pd

from foresight.evaluate import HISTORY_FROM, hits_at_k
from foresight.monitor.cohorts import as_scored
from foresight.monitor.policy import forward, production, shadow
from foresight.monitor.report import CONTENDERS

rows = as_scored(HISTORY_FROM, "2026-03-31")
names = [c.name for c in CONTENDERS]
for label, shift in (("at the end", 0), ("at the notice", 60)):
    early = rows.assign(end_date=rows.end_date
                        - pd.Timedelta(days=shift))
    s = shadow(early, CONTENDERS)
    log, served = forward(s, names[0], names[1:])
    changes = served[served != served.shift()].iloc[1:]
    done = production(s, served)
    done = done[done.moment <= "2025-10-02"]
    hits = sum(hits_at_k(g.not_renewed, g.model, g.contract_id)
               for _, g in done.groupby("moment"))
    print(f"Outcomes known {label}: {hits} leavers in 360 calls")
    for day, name in changes.items():
        print(f"  {day:%d %B}: {name} takes over")
