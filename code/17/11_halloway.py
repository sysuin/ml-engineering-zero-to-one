# Halloway Healthcare left the Midwest on 30 June 2024: what the
# forecasts did, and what they do when they are told.
import pandas as pd

from foresight.forecast.backtest import backtest, by_horizon
from foresight.forecast.baselines import (seasonal_naive,
                                          seasonal_smoothing)
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units, sales)

HALLOWAY = 1                                  # its account_id
key = sales()[sales().key == 1]
key = key.groupby(["month", "region"]).units.sum().unstack()
without = monthly_units(["region"], drop_accounts=(HALLOWAY,))
key.insert(1, "Midwest: Halloway alone",
           monthly_units(["region"]).Midwest - without.Midwest)
halves = {"Jan-Jun": key.loc["2024-01":"2024-06"].mean(),
          "Jul-Dec": key.loc["2024-07":"2024-12"].mean()}
print("Key accounts, units a month, 2024   Jan-Jun   Jul-Dec")
for region in key.columns:
    print(f"  {region:<32}{halves['Jan-Jun'][region]:>8,.0f}"
          f"{halves['Jul-Dec'][region]:>10,.0f}")

# The year after the exit. Every origin in it is after Halloway's
# notice, so a forecaster could have rebuilt the history without it.
year = pd.period_range("2024-07", "2025-06", freq="M")
tail = monthly_units(CATEGORY_REGION, strand="tail")
results = {}
for label, drop in (("as recorded", ()), ("without Halloway",
                                         (HALLOWAY,))):
    units = monthly_units(CATEGORY_REGION, drop_accounts=drop)
    midwest = [c for c in established(units).columns
               if c[1] == "Midwest"]
    model = backtest_model(units, (tail,), targets=year)
    for name, r in (("seasonal naive",
                     backtest(units[midwest], seasonal_naive,
                              targets=year)),
                    ("smoothing", backtest(units[midwest],
                                           seasonal_smoothing,
                                           targets=year)),
                    ("model", model[model.series.isin(midwest)])):
        results[f"{name}, {label}"] = r

print("\nMidwest, four categories, July 2024 to June 2025")
table = by_horizon(results).map("{:.1%}".format)
table["bias"] = [f"{r.forecast.sum() / r.actual.sum() - 1:+.1%}"
                 for r in results.values()]
print(table.rename_axis(None, axis=1).to_string())
