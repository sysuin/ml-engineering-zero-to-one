# Scoring forecasts across many series, and where MAPE goes wrong.
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import legacy_ids
from foresight.forecast.backtest import backtest, mae, mape, wape
from foresight.forecast.baselines import naive, seasonal_naive
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = established(monthly_units(CATEGORY_REGION))   # 20 series
q3 = pd.period_range("2025-07", "2025-09", freq="M")
print(f"{units.shape[1]} series, forecast at the end of June 2025 "
      "for July to September")
print(f"  {'':<16}{'MAE':>8}{'MAPE':>8}{'WAPE':>8}")
for name, rule in (("naive", naive),
                   ("seasonal naive", seasonal_naive)):
    r = pd.concat([backtest(units, rule, horizons=(h,), targets=[t])
                   for h, t in zip((1, 2, 3), q3)])
    print(f"  {name:<16}{mae(r.actual, r.forecast):>8,.0f}"
          f"{mape(r.actual, r.forecast):>8.1%}"
          f"{wape(r.actual, r.forecast):>8.1%}")
by_series = r.groupby("series").apply(
    lambda g: mae(g.actual, g.forecast)).sort_values()
print(f"  seasonal naive MAE by series: {by_series.iloc[0]:,.0f} to "
      f"{by_series.iloc[-1]:,.0f} units")

# One level down: every long-tail account in the Midwest.
with sqlite3.connect(ML_WAREHOUSE) as con:
    orders = pd.read_sql_query("""
        SELECT substr(o.order_date, 1, 7) AS month, o.account_id,
               SUM(l.qty) AS units
        FROM orders o JOIN order_lines l USING (order_id)
        JOIN accounts a USING (account_id)
        WHERE a.region_id = 3 AND a.is_key_account = 0
          AND o.order_date >= '2024-01-01'
        GROUP BY 1, 2""", con)
    orders["account_id"] = orders.account_id.replace(legacy_ids(con))
grid = orders.pivot_table(index="month", columns="account_id",
                          values="units", aggfunc="sum").fillna(0)
last_year = grid.loc[:"2024-12"].to_numpy()
actual = grid.loc["2025-01":].to_numpy()
sold = actual > 0
print(f"\n{grid.shape[1]:,} Midwest long-tail accounts, 2025 by month")
print(f"  {1 - sold.mean():.1%} of account-months bought nothing")
guesses = {"seasonal naive": last_year,
           "2024's average": np.broadcast_to(last_year.mean(axis=0),
                                             actual.shape),
           "zero, always": np.zeros_like(actual)}
print(f"  {'':<16}{'MAPE*':>8}{'WAPE':>8}")
for name, f in guesses.items():
    print(f"  {name:<16}{mape(actual[sold], f[sold]):>8.1%}"
          f"{wape(actual, f):>8.1%}")
print("  * MAPE on the months with a sale; the rest divide by zero")
