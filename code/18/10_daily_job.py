# The daily job on two days, and its false alerts over all of 2024.
from foresight.anomaly import job
from foresight.anomaly.alert import false_alerts, normal_days
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import all_series, tickets
from foresight.anomaly.forest import rolling_ranks, series_days

t = tickets()
for day in ["2024-09-17", "2024-10-07"]:
    alerts = job.run(day, t)
    print(f"{day}: {len(alerts)} alert(s)")
    for a in alerts[:2]:
        print("\n".join(a.lines()))

# The same settings run over every day at once, to count its alerts.
S = all_series(t, job.HISTORY_FROM, "2024-12-31")
chart = control_chart(S, job.WINDOW, method="poisson",
                      alpha=job.ALPHA, skip_alerts=True).alerts
ranks = rolling_ranks(series_days(S, job.WINDOW), "2024-01-01")
forest = (ranks["rank"] <= job.Q).unstack("series", fill_value=False)
both = chart.loc["2024"] | forest.reindex_like(chart.loc["2024"])
normal = normal_days(both.index, "2024-01-01", "2024-12-31",
                     ("2024-10-07", "2024-12-20"))
n, per_month = false_alerts(both, normal)
quiet = both[normal.to_numpy()].any(axis=1).sum()
print(f"\n2024, normal days: {n} alerts on {quiet} days, "
      f"{per_month:.2f} a month")
defect = both.loc["2024-10-07":"2024-12-20"]
pm = "supplier: Pemberton Mills"
print(f"Defect window: Pemberton alerted on {defect[pm].sum()} of "
      f"{len(defect)} days")
