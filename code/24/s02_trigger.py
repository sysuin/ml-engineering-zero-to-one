# Exercise 2: a retraining trigger on the ratio of a list's notices to
# the leavers its chances expected, at three thresholds, beside the
# Poisson check the monitor uses.
import pandas as pd

from foresight.evaluate import HISTORY_FROM, hits_at_k
from foresight.models.logistic import log_loss
from foresight.monitor import watch
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.policy import Contender, record, shadow
from foresight.monitor.report import v06

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
s = watch.scored(v06().fit(train), rows[rows.moment >= GO_LIVE])
e = watch.early(s, "2025-12-31")
e["ratio"] = e.noticed / e.expected
due = s.groupby("moment").end_date.max() - pd.Timedelta(days=59)
e["day"] = due.reindex(e.index)          # the morning after the last
print("Notices over expected, list by list")
for half in (e.ratio.iloc[:5], e.ratio.iloc[5:]):
    print("  " + "  ".join(f"{m:%b} {r:.2f}" for m, r in half.items()))

rules = {"never": ()}
for t in (1.25, 1.5, 2.0):
    rules[f"ratio {t}"] = tuple(e.day[e.ratio >= t])
rules["Poisson"] = tuple(e.day[e.alert])
rules["monthly"] = None
policies = [Contender(k, v06, "monthly" if v is None else
                      "never" if not v else "trigger", v or ())
            for k, v in rules.items()]
sh = known(shadow(rows, policies), "2026-01-01")
print(f"\n{'trigger':<12}{'refits':>7}{'leavers':>9}{'log loss':>10}"
      f"{'average':>9}")
for c in policies:
    r = record(sh, c.name)
    hits = sum(hits_at_k(g.not_renewed, g.model, g.contract_id)
               for _, g in r.groupby("moment"))
    loss = log_loss(r.not_renewed.to_numpy(), r.model.to_numpy())
    print(f"{c.name:<12}{r.fitted_on.nunique() - 1:>7}{hits:>9}"
          f"{loss:>10.4f}{r.model.mean():>9.1%}")
left = sh.not_renewed[sh.name == "never"].mean()
print(f"\nShare that left: {left:.1%}")
