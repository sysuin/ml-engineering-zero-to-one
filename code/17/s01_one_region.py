# Exercise 1: region by region, how far each forecast gets below
# seasonal naive, category x region, 2024-2025, h=1-3 pooled.
from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import (seasonal_naive,
                                          seasonal_smoothing)
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = monthly_units(CATEGORY_REGION)
old = established(units)
model = backtest_model(units, (monthly_units(CATEGORY_REGION, "tail"),))
results = {"seasonal naive": backtest(old, seasonal_naive),
           "smoothing": backtest(old, seasonal_smoothing),
           "model": model[model.series.isin(old.columns)]}
print(f"{'WAPE':<12}" + "".join(f"{k:>16}" for k in results))
for region in sorted({s[1] for s in old.columns}):
    cells = []
    for r in results.values():
        g = r[r.series.map(lambda s: s[1]) == region]
        cells.append(wape(g.actual, g.forecast))
    print(f"  {region:<10}" + "".join(f"{c:>16.1%}" for c in cells))
