# timeout: 300
# Exercise 2: the Poisson chart tuned again with a 28-day baseline.
import numpy as np
import pandas as pd

from foresight.anomaly.alert import (false_alerts, first_alert,
                                     normal_days)
from foresight.anomaly.control import control_chart
from foresight.anomaly.counts import DEFECT_SKU, all_series, tickets
from foresight.config import rng

t = tickets()
start, end = pd.Timestamp("2024-10-07"), pd.Timestamp("2024-12-20")
pm, qual = "supplier: Pemberton Mills", "category: Quality"
S = all_series(t, "2022-01-01", "2025-12-31")
tune = normal_days(S.index, "2023-01-01", "2023-12-31", (start, end))
test = normal_days(S.index, "2024-01-01", "2024-12-31", (start, end))

chosen = {}
print(f"{'window':>6}{'alpha':>8}{'2023':>7}{'2024':>7}"
      f"{'Pemberton':>12}{'Quality':>12}")
for window in (56, 28):
    for alpha in [1e-2, 3e-3, 1e-3, 1e-4, 1e-5]:
        a = control_chart(S, window, alpha=alpha, skip_alerts=True)
        _, t23 = false_alerts(a.alerts, tune)
        if t23 <= 1:
            break
    chosen[window] = alpha
    _, t24 = false_alerts(a.alerts, test)
    p = first_alert(a.alerts[pm], start)
    q = first_alert(a.alerts[qual], start)
    print(f"{window:>6}{alpha:>8g}{t23:>7.2f}{t24:>7.2f}"
          f"{p.strftime('%Y-%m-%d'):>12}{q.strftime('%Y-%m-%d'):>12}")

defect = t.index[(t.sku == DEFECT_SKU) & t.day.between(start, end)]
order = rng().permutation(defect)
print(f"\n{'kept':>6}" + "".join(f"{w:>9}-day" for w in chosen))
for share in [0.2, 0.1, 0.05]:
    keep = order[: int(round(share * len(defect)))]
    thin = all_series(t.drop(index=np.setdiff1d(defect, keep)),
                      "2022-01-01", "2025-12-31")
    cells = []
    for window, alpha in chosen.items():
        a = control_chart(thin, window, alpha=alpha, skip_alerts=True)
        first = first_alert(a.alerts[pm], start)
        late = first is None or first > end
        cells.append("missed" if late else f"{(first - start).days}")
    print(f"{share:>6.0%}" + "".join(f"{c:>13}" for c in cells))
