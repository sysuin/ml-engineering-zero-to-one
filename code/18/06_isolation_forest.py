# Isolation forests on two kinds of row: whole days, and series-days.
import json

import numpy as np
import pandas as pd

from foresight.anomaly.counts import all_series, tickets
from foresight.anomaly.forest import (c, fit_forest, path_lengths,
                                      series_days)

S = all_series(tickets(), "2022-01-01", "2025-12-31")
day, month = pd.Timestamp("2024-10-07"), pd.Timestamp("2024-10-01")
start = month - pd.Timedelta(days=365)     # October's training year
year = (S.index >= start) & (S.index < month)
pm, qual = "supplier: Pemberton Mills", "category: Quality"
print(f"Training year: {S.index[year][0]:%Y-%m-%d} to "
      f"{S.index[year][-1]:%Y-%m-%d}")
print(f"Each tree sees 256 rows; c(256) = {c(256)[0]:.2f}")

# 1. A row is a whole day: 22 columns, one per series.
days = fit_forest(S[year].to_numpy())
normal = path_lengths(days, S[year].to_numpy())
odd = path_lengths(days, S.loc[[day]].to_numpy())[0]
share = (normal <= odd).mean()
print(f"\nRows are days ({S.shape[1]} columns)")
print(f"  path length, training days: median {np.median(normal):.2f},"
      f" shortest {normal.min():.2f}")
print(f"  2024-10-07: {odd:.2f}; {share:.1%} of training days were "
      "as short")

# 2. A row is one series on one day: two columns of surprise.
rows = series_days(S)
d = rows.index.get_level_values("day")
train = rows[(d >= start) & (d < month)]
forest = fit_forest(train.to_numpy())
normal2 = path_lengths(forest, train.to_numpy())
print(f"\nRows are series-days (2 columns, {len(train):,} rows)")
print(f"  path length, training rows: median {np.median(normal2):.2f},"
      f" shortest {normal2.min():.2f}")
print(f"\n  {'row':32}{'today':>7}{'week':>7}{'path':>7}{'score':>7}")
low = train.iloc[np.argsort(normal2)[:3]].index.tolist()
bill = "category: Billing"
for dd, ss in low + [(day, pm), (day, qual), (day, bill)]:
    x = rows.loc[[(dd, ss)]].to_numpy()
    h = path_lengths(forest, x)[0]
    score = 2 ** (-h / c(256)[0])        # what score_samples negates
    print(f"  {dd:%Y-%m-%d} {ss.split(': ')[1]:21}{x[0, 0]:>7.1f}"
          f"{x[0, 1]:>7.1f}{h:>7.2f}{score:>7.2f}")

json.dump({"days": {"normal": np.round(normal, 3).tolist(),
                    "event": round(float(odd), 3)},
           "series_days": {
               "normal": np.round(normal2, 3).tolist(),
               "event": round(float(path_lengths(
                   forest, rows.loc[[(day, pm)]].to_numpy())[0]), 3)},
           "c256": float(c(256)[0])},
          open("code/18/06_isolation_forest.json", "w"))
