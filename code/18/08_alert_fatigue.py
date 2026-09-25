# timeout: 300
# Tune each detector on 2023 to one false alert a month; test on 2024.
import numpy as np
import pandas as pd

from foresight.anomaly.alert import (false_alerts, first_alert,
                                     normal_days)
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import DEFECT_SKU, all_series, tickets
from foresight.anomaly.forest import (forest_alerts, rolling_ranks,
                                      series_days)
from foresight.config import rng

t = tickets()
start, end = pd.Timestamp("2024-10-07"), pd.Timestamp("2024-12-20")
pm = "supplier: Pemberton Mills"
grids = {"mean + k sd": ("k", [3, 4, 5, 6, 8]),
         "Poisson": ("alpha", [1e-2, 3e-3, 1e-3, 1e-4, 1e-5]),
         "forest": ("q", [5e-3, 2e-3, 1e-3, 5e-4, 2e-4])}


def detect(S, family, value, ranks=None):
    """Alerts on every series for one setting of one detector."""
    if family == "mean + k sd":
        return control_chart(S, method="sigma", k=value).alerts
    if family == "Poisson":
        return control_chart(S, method="poisson", alpha=value,
                             skip_alerts=True).alerts
    flags = forest_alerts(ranks, value)
    return flags.unstack("series", fill_value=False)


S = all_series(t, "2022-01-01", "2025-12-31")
ranks = rolling_ranks(series_days(S), "2023-01-01")
tune = normal_days(S.index, "2023-01-01", "2023-12-31", (start, end))
test = normal_days(S.index, "2024-01-01", "2024-12-31", (start, end))

print(f"{'':13}{'setting':<13}{'2023':>6}{'2024':>6}{'caught':>12}")
chosen = {}
for family, (param, values) in grids.items():
    for v in values:                 # loosest first
        a = detect(S, family, v, ranks)
        _, per_month = false_alerts(a, tune)
        _, later = false_alerts(a, test)
        caught = first_alert(a[pm], start)
        pick = per_month <= 1 and family not in chosen
        if pick:
            chosen[family] = v
        print(f"{family if v == values[0] else '':13}"
              f"{param + '=' + format(v, 'g'):<13}{per_month:>6.2f}"
              f"{later:>6.2f}{caught.strftime('%Y-%m-%d'):>12}"
              f"{'  <- chosen' if pick else ''}")

# A quieter defect: keep only part of the tickets that name the cloth
# in the defect window, and ask how long each tuned detector takes.
defect = t.index[(t.sku == DEFECT_SKU) & t.day.between(start, end)]
order = rng().permutation(defect)
print(f"\n{len(defect)} defect tickets, "
      f"{len(defect) / 75:.1f} a day on top of Pemberton's "
      f"{S.loc[:start - pd.Timedelta(days=1), pm].tail(56).mean():.2f}")
print(f"{'kept':>6}{'a day':>7}" + "".join(f"{f:>14}" for f in grids))
for share in [0.5, 0.2, 0.1, 0.05]:
    keep = order[: int(round(share * len(defect)))]
    thin = t.drop(index=np.setdiff1d(defect, keep))
    S2 = all_series(thin, "2022-01-01", "2025-12-31")
    r2 = rolling_ranks(series_days(S2), "2024-10-01")
    cells = []
    for family, v in chosen.items():
        first = first_alert(detect(S2, family, v, r2)[pm], start)
        late = first is None or first > end
        cells.append("missed" if late
                     else f"{(first - start).days} days")
    print(f"{share:>6.0%}{len(keep) / 75:>7.1f}"
          + "".join(f"{c:>14}" for c in cells))
