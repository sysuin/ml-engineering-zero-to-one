# Sanitation, launched in April 2024: forecasting a category with no
# history, then with a little, measured on its first twelve months.
import json

import pandas as pd

from foresight.forecast.backtest import wape
from foresight.forecast.baselines import seasonal_smoothing
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (CATEGORY_REGION, monthly_units,
                                       sales)

units = monthly_units(CATEGORY_REGION)
tail = monthly_units(CATEGORY_REGION, strand="tail")
products = sales().groupby("category").sku.nunique()
others = ["Cleaning", "Facilities", "Packaging", "Safety"]
q2 = {y: units.loc[f"{y}-04":f"{y}-06"].T.groupby(level=0).sum().sum(
      axis=1) for y in (2023, 2024)}
print("Units, April-June        2023      2024    change")
for label, a, b in (("the other four", q2[2023][others].sum(),
                     q2[2024][others].sum()),
                    ("Sanitation", 0, q2[2024]["Sanitation"]),
                    ("all five", q2[2023].sum(), q2[2024].sum())):
    print(f"  {label:<15}{a:>10,.0f}{b:>10,.0f}{b - a:>+10,.0f}")

launch = pd.Period("2024-04", freq="M")
targets = pd.period_range(launch, launch + 11, freq="M")
model = backtest_model(units, (tail,), horizons=(1,), targets=targets)
model = model.set_index(["series", "target"]).forecast
rows = []
for target in targets:
    history = units.loc[:target - 1]
    smooth = seasonal_smoothing(history[others], 1)
    for region in units.columns.get_level_values("region").unique():
        s = ("Sanitation", region)
        per_product = sum(smooth[(c, region)] / products[c]
                          for c in others) / len(others)
        neighbours = per_product * products["Sanitation"]
        last = history[s].iloc[-1]
        last = neighbours if pd.isna(last) else last
        rows.append({"target": target, "actual": units.loc[target, s],
                     "neighbours": neighbours,
                     "as Cleaning": smooth[("Cleaning", region)],
                     "naive": last,
                     "model": model.get((s, target), last)})
d = pd.DataFrame(rows)
d["age"] = (d.target - launch).map(lambda x: x.n) + 1
rules = ["neighbours", "as Cleaning", "naive", "model"]

first = d[d.age == 1][["actual"] + rules[:2]].sum()
print(f"\nLaunch month, five regions: actual {first.actual:,.0f}")
for rule in rules[:2]:
    print(f"  {rule:<14}{first[rule]:>8,.0f}"
          f"  ({first[rule] / first.actual - 1:+.1%})")
print("\nWAPE, h=1       " + "".join(f"{r:>13}" for r in rules))
for label, part in (("months 1-3", d.age <= 3),
                    ("months 4-12", d.age > 3),
                    ("all twelve", d.age > 0)):
    g = d[part]
    print(f"  {label:<14}" + "".join(
        f"{wape(g.actual, g[r]):>13.1%}" for r in rules))
print("  (seasonal naive has no forecast until April 2025;"
      " the model's\n   first is month 4, and naive fills in before)")

monthly = d.groupby("target")[["actual"] + rules].sum()
with open("code/17/08_cold_start.json", "w") as f:
    json.dump({"months": [str(m) for m in monthly.index]}
              | {k: monthly[k].round(0).tolist() for k in monthly}, f)
