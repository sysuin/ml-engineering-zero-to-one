# Exercise 4: choose the smoothing constants on 2024, then score the
# choice once on 2025 beside the defaults and seasonal naive.
import pandas as pd

from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import (seasonal_naive,
                                          seasonal_smoothing)
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = established(monthly_units(CATEGORY_REGION))
years = {y: pd.period_range(f"{y}-01", f"{y}-12", freq="M")
         for y in (2024, 2025)}


def score(alpha, gamma, year):
    r = backtest(units, lambda hist, h: seasonal_smoothing(
        hist, h, alpha=alpha, gamma=gamma), targets=years[year])
    return wape(r.actual, r.forecast)


grid = {(a, g): score(a, g, 2024) for a in (0.1, 0.2, 0.3, 0.5)
        for g in (0.1, 0.2, 0.3, 0.5)}
best = min(grid, key=grid.get)
print(f"best on 2024: alpha {best[0]}, gamma {best[1]} "
      f"({grid[best]:.1%}; the defaults {grid[(0.3, 0.2)]:.1%})")
naive = backtest(units, seasonal_naive, targets=years[2025])
print("2025, h=1-3 pooled")
print(f"  chosen on 2024   {score(*best, 2025):.1%}")
print(f"  defaults         {score(0.3, 0.2, 2025):.1%}")
print(f"  seasonal naive   {wape(naive.actual, naive.forecast):.1%}")
