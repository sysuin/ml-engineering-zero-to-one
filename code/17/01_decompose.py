# One series taken apart into trend, season and what is left over.
import json

import numpy as np
import pandas as pd

from foresight.forecast.series import monthly_units, sales

# Why the panels start in 2023: the record of the key accounts does.
d = sales()
cleaning = d[d.category == "Cleaning"]
split = cleaning.groupby(["month", "key"]).units.sum().unstack()
print("Cleaning units      long tail   key accounts")
for m in ("2022-11", "2022-12", "2023-01", "2023-02"):
    tail, key = split.loc[m]
    key = "not on record" if np.isnan(key) else f"{key:,.0f}"
    print(f"  {m}           {tail:>9,.0f}   {key:>13}")

y = monthly_units(["category"])["Cleaning"]       # 2023 onwards
weights = np.r_[0.5, np.ones(11), 0.5] / 12       # a centred year
trend = y.rolling(13, center=True).apply(
    lambda w: w @ weights, raw=True)
ratio = y / trend
season = ratio.groupby(ratio.index.month).mean()
season = season / season.mean()                   # average to 1
seasonal = pd.Series(season.loc[y.index.month].to_numpy(), y.index)
remainder = y / (trend * seasonal)

print("\nSeasonal index, Cleaning, all regions")
for half in (range(1, 7), range(7, 13)):
    print("  month " + "".join(f"{m:>7}" for m in half))
    print("  index " + "".join(f"{season[m]:>7.2f}" for m in half))
print("\nTrend (centred 12-month average)")
for m in ("2023-07", "2024-01", "2024-03", "2024-06", "2024-09",
          "2025-06"):
    print(f"  {m}  {trend[m]:>8,.0f}")
spread = remainder.dropna()
print(f"\nRemainder: {len(spread)} months, typical size "
      f"{(spread - 1).abs().mean():.1%}, largest "
      f"{(spread - 1).abs().max():.1%}")

parts = {"units": y, "trend": trend.round(1),
         "seasonal": seasonal.round(4), "remainder": remainder.round(4)}
with open("code/17/01_decompose.json", "w") as f:
    json.dump({"months": [str(m) for m in y.index]}
              | {k: [None if np.isnan(v) else v for v in s]
                 for k, s in parts.items()}, f)
