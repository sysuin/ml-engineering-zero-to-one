# Exercise 3: tenure drifts all year and the list does not care. Where
# its PSI comes from, and what the model does with it.
from foresight.evaluate import HISTORY_FROM
from foresight.monitor.alerts import effects
from foresight.monitor.cohorts import GO_LIVE, as_scored, known
from foresight.monitor.psi import Baseline
from foresight.monitor.report import v06

rows = as_scored(HISTORY_FROM, "2026-03-31")
rows["tenure_months"] = rows.tenure_days // 30
rows["tenure_years"] = rows.tenure_days // 365
rows["past_a_year"] = rows.tenure_days % 365
train = known(rows[rows.end_date >= HISTORY_FROM], GO_LIVE)
year = rows[rows.moment >= GO_LIVE]
cols = ["tenure_days", "tenure_months", "tenure_years", "past_a_year"]
base = Baseline(train, cols)
print(f"{'':<16}{'training':>10}{'2025':>8}{'PSI':>7}")
for c in cols:
    print(f"{c:<16}{train[c].median():>10.0f}{year[c].median():>8.0f}"
          f"{base.column(year, c):>7.2f}")
model = v06().fit(train)
w = model.model_.weights()["tenure_days"] * model.map_.a_
print(f"\nv0.6's weight on tenure: {w:+.4f} per standard deviation")
print(f"The drift's effect on the average log-odds:"
      f" {effects(model, train, year)['tenure_days']:+.4f}")
