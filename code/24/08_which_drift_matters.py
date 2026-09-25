# Which drift matters: each column's PSI over 2025's lists beside how
# far its drift moved v0.6's average log-odds, which is the column's
# average part in explain.contributions(), now minus in training.
import numpy as np

from foresight.evaluate import HISTORY_FROM
from foresight.explain import contributions, log_odds
from foresight.monitor.alerts import EFFECT, input_drift
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.psi import ACT
from foresight.monitor.report import v06
from foresight.train import COLUMNS

rows = as_scored(HISTORY_FROM, "2026-03-31")
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
model = v06().fit(train)
year = rows[rows.moment >= GO_LIVE].assign(moment=GO_LIVE)
d = input_drift(model, train, year, COLUMNS + ["supplier_voss"])
w = model.model_.weights() * model.map_.a_     # per sd, after the map
source = w.index.str.replace("log_", "").str.split("=").str[0]
d["weight"] = [np.abs(w[source == c]).max() if c in COLUMNS else 0.0
               for c in d.column]
print(f"All {len(year):,} contracts listed in 2025 against v0.6's"
      f" {len(train):,}")
print(f"{'column':<18}{'PSI':>7}{'largest weight':>16}"
      f"{'effect':>9}  verdict")
for r in d.sort_values("psi", ascending=False).itertuples():
    moved, big = r.psi >= ACT, abs(r.effect) >= EFFECT
    verdict = ("alert" if moved and big else "report" if moved
               else "moves score" if big else "")
    print(f"{r.column:<18}{r.psi:>7.2f}{r.weight:>16.3f}"
          f"{r.effect:>+9.3f}  {verdict}")
shift = (log_odds(model.predict_proba(year)).mean()
         - log_odds(model.predict_proba(train)).mean())
parts = (contributions(model, year)[1].mean()
         - contributions(model, train)[1].mean())
print(f"\nAverage log-odds, 2025 minus training: {shift:+.3f}"
      f"\nThe columns' parts, summed: {parts.sum():+.3f}")
