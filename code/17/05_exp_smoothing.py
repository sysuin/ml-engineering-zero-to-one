# Exponential smoothing: a level that learns, and a season that does.
import pandas as pd

from foresight.forecast.backtest import backtest, by_horizon
from foresight.forecast.baselines import (naive, seasonal_naive,
                                          seasonal_smoothing,
                                          simple_smoothing)
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = established(monthly_units(CATEGORY_REGION))
series = ("Safety", "Southwest")
origin = pd.Period("2025-06", freq="M")
history = units.loc[:origin]
print(f"{series[0]} in the {series[1]}: {origin + 1}, "
      f"forecast at the end of {origin}")
for name, f in (("seasonal naive", seasonal_naive),
                ("simple smoothing", simple_smoothing),
                ("seasonal smoothing", seasonal_smoothing)):
    print(f"  {name:<20}{f(history, 1)[series]:>8,.0f}")
print(f"  {'actual':<20}{units.loc[origin + 1, series]:>8,.0f}")

results = {"naive": backtest(units, naive),
           "seasonal naive": backtest(units, seasonal_naive),
           "simple smoothing": backtest(units, simple_smoothing),
           "seasonal smoothing": backtest(units, seasonal_smoothing)}
print("\nWAPE, 2024-2025, 20 category x region series")
table = by_horizon(results).map("{:.1%}".format)
print(table.rename_axis(None, axis=1).to_string())

print("\nSeasonal smoothing with other settings (not chosen from this)")
for alpha, gamma in ((0.1, 0.1), (0.3, 0.2), (0.5, 0.5)):
    r = backtest(units, lambda hist, h: seasonal_smoothing(
        hist, h, alpha=alpha, gamma=gamma))
    w = by_horizon({"": r}).iloc[0]
    print(f"  alpha {alpha}, gamma {gamma}: " +
          "  ".join(f"{v:.1%}" for v in w))
