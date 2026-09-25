# 80% intervals from two quantile models, checked against what
# happened, then widened from their own record.
import json

import pandas as pd

from foresight.forecast.intervals import coverage, widen
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (PRODUCT_REGION, established,
                                       monthly_units)

units = monthly_units(PRODUCT_REGION)
tail = monthly_units(PRODUCT_REGION, strand="tail")
old = established(units).columns
r = backtest_model(units, (tail,), alphas=(None, 0.1, 0.9))
r = widen(r[r.series.isin(old)])
r["year"] = r.target.dt.year


def table(title, lo, hi, rows):
    print(title)
    print("          h=1     h=2     h=3   width")
    for year, g in rows.groupby("year"):
        cells = [coverage(k, lo, hi) for _, k in g.groupby("h")]
        width = ((g[hi] - g[lo]) / g.forecast).median()
        print(f"  {year}" + "".join(f"{c:>8.1%}" for c in cells)
              + f"{width:>8.1%}")


table("Share of actuals inside the 80% interval, product x region",
      "q10", "q90", r)
for year, g in r.groupby("year"):
    print(f"  {year}: below q10 {(g.actual < g.q10).mean():.1%}, "
          f"above q90 {(g.actual > g.q90).mean():.1%}")
table("\nWidened from earlier misses", "lo", "hi", r.dropna())

series = ("Packaging", "MRD-PAC-020", "Southwest")
origin = pd.Period("2025-09", freq="M")
fan = r[(r.series == series) & (r.origin == origin)]
print(f"\n{series[1]}, {series[2]}, forecast at the end of {origin}")
print("           q10  median     q90   actual   widened")
for _, x in fan.iterrows():
    print(f"  {x.target}{x.q10:>7,.0f}{x.forecast:>8,.0f}{x.q90:>8,.0f}"
          f"{x.actual:>9,.0f}   {x.lo:,.0f}-{x.hi:,.0f}")
history = units.loc["2024-01":origin, series]
with open("code/17/09_quantiles.json", "w") as f:
    json.dump({"history": {str(m): v for m, v in history.items()},
               "fan": [{"target": str(x.target)} | {
                   k: round(x[k], 1) for k in
                   ("actual", "q10", "forecast", "q90", "lo", "hi")}
                   for _, x in fan.iterrows()]}, f)
