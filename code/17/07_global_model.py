# One LightGBM model for every product in every region, backtested
# beside the rules it has to beat.
import json

from foresight.forecast.backtest import backtest, by_horizon, wape
from foresight.forecast.baselines import (naive, seasonal_naive,
                                          seasonal_smoothing)
from foresight.forecast.model import backtest_model, training_rows
from foresight.forecast.series import (PRODUCT_REGION, established,
                                       monthly_units)

units = monthly_units(PRODUCT_REGION)
tail = monthly_units(PRODUCT_REGION, strand="tail")
old = established(units)
rows, train = training_rows(units, (tail,))
print(f"{units.shape[1]} product x region series; {old.shape[1]} on "
      "sale since 2023 are scored")
print(f"learning rows: {len(rows):,} from 2023 on, "
      f"{len(train) - len(rows):,} more from the long tail")

results = {"naive": backtest(old, naive),
           "seasonal naive": backtest(old, seasonal_naive),
           "seasonal smoothing": backtest(old, seasonal_smoothing)}
for name, extra in (("model, 2023 on", ()),
                    ("model + long tail", (tail,))):
    r = backtest_model(units, extra)
    results[name] = r[r.series.isin(old.columns)]
print("\nWAPE, 2024-2025")
table = by_horizon(results).map("{:.1%}".format)
print(table.rename_axis(None, axis=1).to_string())

one = {k: r[r.h == 1] for k, r in results.items()}
print("\nh=1 by year      seasonal naive   smoothing   model")
for year in (2024, 2025):
    cells = [wape(g.actual, g.forecast) for g in
             (one[k][one[k].target.dt.year == year] for k in
              ("seasonal naive", "seasonal smoothing",
               "model + long tail"))]
    print(f"  {year}          {cells[0]:>14.1%}{cells[1]:>12.1%}"
          f"{cells[2]:>8.1%}")

per_series = {k: one[k].groupby("series").apply(
    lambda g: wape(g.actual, g.forecast)) for k in one}
model = per_series["model + long tail"]
for rival in ("seasonal naive", "seasonal smoothing"):
    wins = (model < per_series[rival]).sum()
    print(f"model beats {rival} on {wins} of {len(model)} series (h=1)")

m = one["model + long tail"].groupby("target")[["forecast",
                                                "actual"]].sum()
bias = m.forecast / m.actual - 1
print("model's total forecast against what sold, h=1:")
print(f"  April 2024 {bias['2024-04']:+.1%}, "
      f"April 2025 {bias['2025-04']:+.1%}")

by_month = {k: one[k].groupby("target").apply(
    lambda g: wape(g.actual, g.forecast)) for k in one}
with open("code/17/07_global_model.json", "w") as f:
    json.dump({"months": [str(m) for m in by_month["naive"].index]}
              | {k: v.round(4).tolist() for k, v in by_month.items()},
              f)
