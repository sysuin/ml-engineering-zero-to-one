# What ordering to each forecast would have cost the buyers in 2025,
# in dollars of surplus stock and lost sales, one month ahead.
# timeout: 300
import json

import pandas as pd

from foresight.forecast.backtest import backtest
from foresight.forecast.baselines import seasonal_naive
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (PRODUCT_REGION, established,
                                       monthly_units)
from foresight.impact.inventory import StockCosts, cost, prices

units = monthly_units(PRODUCT_REGION)
tail = monthly_units(PRODUCT_REGION, strand="tail")
old = established(units)
year = pd.period_range("2025-01", "2025-12", freq="M")
model = backtest_model(units, (tail,), alphas=(None, 0.9),
                       horizons=(1,), targets=year)
model = model[model.series.isin(old.columns)].reset_index(drop=True)
naive = backtest(old, seasonal_naive, horizons=(1,), targets=year)
# A fair rival for the 90th percentile: seasonal naive raised by the
# ratio its 2024 forecasts fell short by, one month in ten.
last_year = pd.period_range("2024-01", "2024-12", freq="M")
before = backtest(old, seasonal_naive, horizons=(1,),
                  targets=last_year)
before = before[before.forecast > 0]
lift = (before.actual / before.forecast).quantile(0.9)
naive["padded"] = naive.forecast * lift
priced = prices()

assumed = StockCosts()
print(f"Assumed: holding a unit costs {assumed.holding:.0%} of its"
      f" cost a year; a surplus")
print(f"waits {assumed.months_held:.0f} month; {assumed.lost:.0%} of a"
      f" shortfall is lost, at list margin.")
print(f"{old.shape[1]} product x region series, 2025, one month"
      " ahead.\n")
orders = [("seasonal naive", naive, "forecast"),
          ("model, median", model, "forecast"),
          (f"seasonal naive x {lift:.2f}", naive, "padded"),
          ("model, 90th percentile", model, "q90")]
print(f"{'order to':<24}{'surplus':>10}{'shortfall':>11}{'total':>11}")
out = {"sensitivity": []}
for name, rec, col in orders:
    c = cost(rec, col, assumed, priced)
    out[name] = c.to_dict()
    print(f"{name:<24}" + "".join(f"{v:>11,.0f}" for v in c)[1:])

print("\nWhat the model saves, as the assumptions move")
print(f"{'surplus waits, share lost':<28}{'point':>10}{'range':>10}")
for months, lost in [(1, 0.25), (1, 0.5), (1, 1.0), (3, 0.5)]:
    c = StockCosts(months_held=months, lost=lost)
    point = (cost(naive, "forecast", c, priced)["total"]
             - cost(model, "forecast", c, priced)["total"])
    band = (cost(naive, "padded", c, priced)["total"]
            - cost(model, "q90", c, priced)["total"])
    out["sensitivity"].append([months, lost, point, band])
    label = f"{months} month{'s' * (months > 1)}, {lost:.0%}"
    print(f"{label:<28}{point:>10,.0f}{band:>10,.0f}")
with open("code/25/09_forecast_value.json", "w") as f:
    json.dump(out, f)
