# Three forecasts that need no model, made at the end of June 2025.
import pandas as pd

from foresight.forecast.baselines import (moving_average, naive,
                                          seasonal_naive)
from foresight.forecast.series import CATEGORY_REGION, monthly_units

units = monthly_units(CATEGORY_REGION)
mw = units[("Cleaning", "Midwest")]
q3 = {y: mw.loc[f"{y}-07":f"{y}-09"].sum() for y in (2023, 2024)}
print(f"Cleaning, Midwest, July-September: 2023 {q3[2023]:,.0f}, "
      f"2024 {q3[2024]:,.0f} ({q3[2024] / q3[2023] - 1:+.1%})\n")
series = ("Safety", "Southwest")
origin = pd.Period("2025-06", freq="M")
history = units.loc[:origin]
rules = {"naive": naive, "seasonal naive": seasonal_naive,
         "moving average": moving_average}

print(f"{series[0]} in the {series[1]}, the last months known:")
for m, v in history[series].iloc[-4:].items():
    print(f"  {m}  {v:>7,.0f}")
print(f"\nForecasts made at the end of {origin}")
print(f"  {'':<8}{'actual':>8}" + "".join(f"{r:>16}" for r in rules))
for h in (1, 2, 3):
    target = origin + h
    cells = "".join(f"{f(history, h)[series]:>16,.0f}"
                    for f in rules.values())
    actual = units.loc[target, series]
    print(f"  {str(target):<8}{actual:>8,.0f}{cells}")
