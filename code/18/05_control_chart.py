# Daily tickets by supplier and category, watched by two control charts.
import json

from foresight.anomaly.alert import false_alerts, normal_days
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import all_series, tickets

t = tickets()
named = t.sku.notna().sum()
print(f"{len(t):,} tickets; {named:,} name a product, "
      f"{named - t.body.str.contains('MRD-').sum():,} only in lower "
      "case")

S = all_series(t, "2022-01-01", "2025-12-31")
sup = S.columns.str.startswith("supplier").sum()
print(f"{S.shape[1]} series: {sup}"
      f" suppliers, {S.columns.str.startswith('category').sum()} "
      f"categories, {len(S):,} days")
normal = normal_days(S.index, "2023-01-01", "2024-12-31",
                     ("2024-10-07", "2024-12-20"))

charts = {"mean + 3 sd": control_chart(S, method="sigma", k=3),
          "Poisson": control_chart(S, method="poisson")}
print(f"\nFalse alerts on {normal.sum()} normal days, 2023-2024")
print(f"{'':14}{'alerts':>8}{'a month':>9}{'from series < 1/day':>21}")
quiet = S.columns[S.mean() < 1]
for name, ch in charts.items():
    n, per_month = false_alerts(ch.alerts, normal)
    few = ch.alerts.loc[normal.to_numpy(), quiet].sum().sum()
    print(f"{name:14}{n:>8}{per_month:>9.2f}{few / max(n, 1):>21.0%}")

pm = "supplier: Pemberton Mills"
print(f"\n{pm}, trailing 56-day baseline")
print(f"{'day':12}{'count':>6}{'mean':>7}{'3 sd limit':>12}"
      f"{'Poisson limit':>15}")
for day in ["2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07",
            "2024-10-08"]:
    sd, po = charts["mean + 3 sd"], charts["Poisson"]
    print(f"{day:12}{S.loc[day, pm]:>6}{sd.centre.loc[day, pm]:>7.2f}"
          f"{sd.upper.loc[day, pm]:>12.2f}"
          f"{po.upper.loc[day, pm]:>15.0f}")

fig = {"days": [d.strftime("%Y-%m-%d") for d in S.loc[
    "2024-07-01":"2025-01-31"].index]}
for skip in (False, True):
    ch = control_chart(S, method="poisson", skip_alerts=skip)
    part = slice("2024-07-01", "2025-01-31")
    fig[f"skip_{skip}"] = {"centre": ch.centre.loc[part, pm].tolist(),
                           "upper": ch.upper.loc[part, pm].tolist()}
fig["counts"] = S.loc["2024-07-01":"2025-01-31", pm].tolist()
json.dump(fig, open("code/18/05_control_chart.json", "w"))
