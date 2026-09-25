# Exercise 3: forecast a product the model never trained on. Safety
# item 5 is removed from the learning rows in every region; its own
# recent months still feed its features. 2025, h=1.
import pandas as pd

from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import seasonal_naive
from foresight.forecast.model import forecast_from, training_rows
from foresight.forecast.series import PRODUCT_REGION, monthly_units

SKU = "MRD-SAF-015"
units = monthly_units(PRODUCT_REGION)
tail = monthly_units(PRODUCT_REGION, strand="tail")
rows, train = training_rows(units, (tail,), horizons=(1,))
year = pd.period_range("2025-01", "2025-12", freq="M")
mine = rows[(rows.sku == SKU) & rows.target.isin(year)]
out = {}
for label, learn in (("trained with it", train),
                     ("never seen", train[train.sku != SKU])):
    out[label] = pd.concat(forecast_from(t - 1, mine, learn)
                           for t in year)
cols = [c for c in units.columns if c[1] == SKU]
out["seasonal naive"] = backtest(units[cols], seasonal_naive,
                                 horizons=(1,), targets=year)
print(f"{SKU}, five regions, 2025, h=1")
for label, r in out.items():
    print(f"  {label:<18}WAPE {wape(r.actual, r.forecast):.1%}")
