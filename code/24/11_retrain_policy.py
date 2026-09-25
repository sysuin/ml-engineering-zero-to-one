# Three retraining policies for v0.6 through 2025, each refit using
# only the outcomes on record that morning: never, every month, and on
# the days the monitor's notice or outcome checks fired.
# timeout: 300
from foresight.evaluate import HISTORY_FROM, hits_at_k
from foresight.models.logistic import log_loss
from foresight.monitor.alerts import year
from foresight.monitor.cohorts import (GO_LIVE, RECORD_ENDS, as_scored,
                                       known)
from foresight.monitor.policy import Contender, paired, record, shadow
from foresight.monitor.report import v06
from foresight.monitor.watch import scored
from foresight.train import COLUMNS

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
model = v06().fit(train)
s = scored(model, rows[rows.moment >= GO_LIVE])
fired = sorted({a.day for a in year((model, train), s, COLUMNS,
                                    GO_LIVE, RECORD_ENDS)
                if a.check in ("notices", "outcomes")})
print("Trigger days: " + ", ".join(f"{d:%d %b}" for d in fired))

policies = [Contender("never", v06, "never"),
            Contender("monthly", v06, "monthly"),
            Contender("trigger", v06, "trigger", tuple(fired))]
sh = known(shadow(rows, policies), "2026-01-01")
never = record(sh, "never")
print(f"\n{len(never):,} contracts in {never.moment.nunique()} lists"
      f" with outcomes; {never.not_renewed.mean():.1%} left")
print(f"{'policy':<10}{'models':>7}{'leavers':>9}{'log loss':>10}"
      f"{'average chance':>16}")
for c in policies:
    r = record(sh, c.name)
    hits = sum(hits_at_k(g.not_renewed, g.model, g.contract_id)
               for _, g in r.groupby("moment"))
    loss = log_loss(r.not_renewed.to_numpy(), r.model.to_numpy())
    print(f"{c.name:<10}{r.fitted_on.nunique():>7}{hits:>9}"
          f"{loss:>10.4f}{r.model.mean():>16.1%}")

print("\nMinus never, paired over the same contracts")
print(f"{'policy':<10}{'leavers per call, points':>26}"
      f"{'AUC within lists':>26}")
for c in policies[1:]:
    d = paired(record(sh, c.name), never)
    p, a = d["precision"], d["auc"]
    print(f"{c.name:<10}{p[0] * 100:>+8.1f} ({p[1] * 100:+.1f} to"
          f" {p[2] * 100:+.1f}){a[0]:>+10.3f} ({a[1]:+.3f} to"
          f" {a[2]:+.3f})")
