# The baselines judged over two years: every month a target, three
# horizons, the window of history growing as the origin moves.
from foresight.forecast.backtest import backtest, by_horizon, wape
from foresight.forecast.baselines import (moving_average, naive,
                                          seasonal_naive)
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = established(monthly_units(CATEGORY_REGION))
results = {"naive": backtest(units, naive),
           "seasonal naive": backtest(units, seasonal_naive),
           "moving average": backtest(units, moving_average)}
r = results["seasonal naive"]
print(f"{r.target.nunique()} target months x {units.shape[1]} series "
      f"x {r.h.nunique()} horizons = {len(r):,} forecasts per rule")
print(f"origins from {r.origin.min()} to {r.origin.max()}\n")
print("WAPE, 2024-2025")
table = by_horizon(results).map("{:.1%}".format)
print(table.rename_axis(None, axis=1).to_string())

print("\nWAPE by year, h=1")
for name, r in results.items():
    one = r[r.h == 1]
    years = one.groupby(one.target.dt.year).apply(
        lambda g: wape(g.actual, g.forecast))
    print(f"  {name:<16}" + "   ".join(f"{y}: {v:.1%}"
                                    for y, v in years.items()))

monthly = results["seasonal naive"].query("h == 1").groupby(
    "target").apply(lambda g: wape(g.actual, g.forecast))
print("\nSeasonal naive, h=1, scored one target month at a time")
print(f"  best  {monthly.idxmin()}  {monthly.min():.1%}")
print(f"  worst {monthly.idxmax()}  {monthly.max():.1%}")
