# Exercise 1: tickets in the last 90 days, summarised three ways.
import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
s = train.tickets_90d
q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
fence = q3 + 1.5 * (q3 - q1)
counts = np.bincount(s)
print(f"Histogram, one bar per count 0 to 4: {counts[:5].tolist()}")
print(f"  the bar at 0 holds {counts[0] / len(s):.1%}")
print(f"Box: quartiles {q1:.0f}, {median:.0f}, {q3:.0f}; whisker"
      f" {fence:.0f}; {(s > fence).sum():,} rows beyond it"
      f" ({(s > fence).mean():.1%})")
print("ECDF and the rate of leaving, by count")
for k in range(4):
    rows = train[s == k]
    print(f"  {k} ticket{'s' if k != 1 else ' '}: {(s <= k).mean():6.1%} at or below,"
          f" {len(rows):>5,} rows, {rows.not_renewed.mean():5.1%} left")
more = train[s >= 4]
print(f"  4 or more: {len(more)} rows, {more.not_renewed.sum()} left")
