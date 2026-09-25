# Forecasting as regression: a row of lag features, and the leak that
# comes from building them relative to the target, not the origin.
import pandas as pd

from foresight.forecast.backtest import TARGETS, by_horizon
from foresight.forecast.model import (FEATURES, GlobalModel,
                                      backtest_model, training_rows)
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = monthly_units(CATEGORY_REGION)
tail = monthly_units(CATEGORY_REGION, strand="tail")
rows, train = training_rows(units, (tail,))
row = rows[(rows.series == ("Safety", "Southwest")) & (rows.h == 3)
           & (rows.origin == pd.Period("2025-06", freq="M"))].iloc[0]
print(f"One row: Safety, Southwest, made at {row.origin} for "
      f"{row.target} (h=3)")
for name in FEATURES:
    value = row[name]
    shown = f"{value:.3f}" if isinstance(value, float) else value
    print(f"  {name:<16}{shown}")
print(f"  {'scale':<16}{row.scale:,.0f}   target {row.actual:,.0f}"
      f"  (ratio {row.actual / row.scale:.3f})")

# The leak: one row per target, lags counted back from the target.
# For h=3 its "last month" is two months after the origin.
leaky_rows, leaky_train = training_rows(units, (tail,), horizons=(1,))
out = []
for h in (1, 2, 3):
    for target in TARGETS:
        seen = leaky_train[leaky_train.target <= target - h]
        test = leaky_rows[leaky_rows.target == target].copy()
        test["forecast"] = GlobalModel().fit(seen).predict(test)
        out.append(test.assign(h=h))
keep = established(units).columns
leaky = pd.concat(out)
apart = pd.concat([backtest_model(units, (tail,), horizons=(h,))
                   for h in (1, 2, 3)])
joint = backtest_model(units, (tail,))
results = {"lags from the target": leaky,
           "from the origin, by h": apart,
           "from the origin, one": joint}
results = {k: r[r.series.isin(keep)] for k, r in results.items()}
print("\nWAPE, 2024-2025, 20 category x region series")
table = by_horizon(results).map("{:.1%}".format)
print(table.rename_axis(None, axis=1).to_string())
