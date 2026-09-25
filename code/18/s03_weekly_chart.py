# timeout: 300
# Exercise 4: a Poisson chart on seven-day totals instead of days.
import numpy as np
import pandas as pd
from scipy.stats import poisson

from foresight.anomaly.alert import first_alert, normal_days
from foresight.anomaly.counts import DEFECT_SKU, all_series, tickets
from foresight.config import rng

t = tickets()
start, end = pd.Timestamp("2024-10-07"), pd.Timestamp("2024-12-20")
pm = "supplier: Pemberton Mills"


def weekly_alerts(S, alpha, window=56):
    """Alert when the last 7 days' total is improbable for a Poisson
    series with 7 times the daily mean of the 56 days before them."""
    week = S.rolling(7).sum()
    mean = S.shift(7).rolling(window, min_periods=window).mean()
    lam = 7 * np.maximum(mean, 1.0 / window)
    limit = pd.DataFrame(poisson.isf(alpha, lam), index=S.index,
                         columns=S.columns).where(mean.notna())
    return (week > limit) & limit.notna()


def episodes_a_month(flags, normal):
    """New alerts only: a series alerting on consecutive days is one
    episode, since a seven-day total stays high for a week."""
    new = flags & ~flags.shift(1, fill_value=False)
    n = new[normal.to_numpy()].sum().sum()
    return n / (normal.sum() / 30.4375)


S = all_series(t, "2022-01-01", "2025-12-31")
tune = normal_days(S.index, "2023-01-01", "2023-12-31", (start, end))
test = normal_days(S.index, "2024-01-01", "2024-12-31", (start, end))
print(f"{'alpha':>8}{'2023':>7}{'2024':>7}   first alert")
for alpha in [1e-3, 1e-4, 1e-5, 1e-6]:
    a = weekly_alerts(S, alpha)
    per_2023 = episodes_a_month(a, tune)
    first = first_alert(a[pm], start)
    print(f"{alpha:>8g}{per_2023:>7.2f}"
          f"{episodes_a_month(a, test):>7.2f}   {first:%Y-%m-%d}")
    if per_2023 <= 1:
        break

defect = t.index[(t.sku == DEFECT_SKU) & t.day.between(start, end)]
order = rng().permutation(defect)
print(f"\n{'kept':>6}{'a day':>7}  days to first weekly alert")
for share in [0.2, 0.1, 0.05]:
    keep = order[: int(round(share * len(defect)))]
    thin = all_series(t.drop(index=np.setdiff1d(defect, keep)),
                      "2022-01-01", "2025-12-31")
    first = first_alert(weekly_alerts(thin, alpha)[pm], start)
    late = first is None or first > end
    print(f"{share:>6.0%}{len(keep) / 75:>7.1f}  "
          f"{'missed' if late else (first - start).days}")
