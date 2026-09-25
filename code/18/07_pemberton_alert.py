# Five detectors on one set of series: when each caught Pemberton.
import pandas as pd

from foresight.anomaly.alert import (MONTH_DAYS, Alert, evidence,
                                     false_alerts, first_alert,
                                     normal_days)
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import all_series, tickets
from foresight.anomaly.forest import (forest_alerts, rolling_ranks,
                                      series_days)

t = tickets()
S = all_series(t, "2022-01-01", "2025-12-31")
start, end = pd.Timestamp("2024-10-07"), pd.Timestamp("2024-12-20")
normal = normal_days(S.index, "2023-01-01", "2024-12-31", (start, end))
pm = "supplier: Pemberton Mills"

# Each detector's alerts on the Pemberton series (the day forest has
# no series, so its alerts are whole days) and on every series.
found = {}
for name, kw in [("mean + 3 sd", dict(method="sigma")),
                 ("Poisson", dict(method="poisson")),
                 ("Poisson, alerts kept out", dict(
                     method="poisson", skip_alerts=True))]:
    a = control_chart(S, **kw).alerts
    found[name] = (a[pm], a)
q_day = 1 / MONTH_DAYS                     # one day a month
day_flags = forest_alerts(rolling_ranks(S, "2023-01-01"), q_day)
found["forest, rows are days"] = (day_flags, day_flags)
q_row = 1 / (MONTH_DAYS * S.shape[1])      # one series-day a month
row_flags = forest_alerts(
    rolling_ranks(series_days(S), "2023-01-01"), q_row)
found["forest, series-days"] = (
    row_flags.xs(pm, level="series"), row_flags)

print(f"Pemberton defect: {start:%Y-%m-%d} to {end:%Y-%m-%d}, "
      f"{(end - start).days + 1} days")
print(f"{'':25}{'first':>12}{'delay':>6}{'days':>6}"
      f"{'false a month':>15}")
for name, (flags, every) in found.items():
    first = first_alert(flags, start)
    delay = (first - start).days
    inside = (flags.index >= start) & (flags.index <= end)
    days = int(flags[inside].sum())
    _, per_month = false_alerts(every, normal)
    print(f"{name:25}{first:%Y-%m-%d}{delay:>6}{days:>6}"
          f"{per_month:>15.2f}")

chart = control_chart(S, method="poisson")
alert = Alert(start, pm, "Poisson", int(S.loc[start, pm]),
              chart.centre.loc[start, pm],
              limit=chart.upper.loc[start, pm],
              evidence=evidence(t, start, pm))
print()
print("\n".join(alert.lines()))
