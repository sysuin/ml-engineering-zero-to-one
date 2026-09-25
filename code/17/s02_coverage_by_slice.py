# Exercise 2: do the 80% intervals hold for every region and every
# category, or only on average? Category x region, 2025, h=1-3.
from foresight.forecast.intervals import coverage
from foresight.forecast.model import backtest_model
from foresight.forecast.series import (CATEGORY_REGION, established,
                                       monthly_units)

units = monthly_units(CATEGORY_REGION)
r = backtest_model(units, (monthly_units(CATEGORY_REGION, "tail"),),
                   alphas=(None, 0.1, 0.9))
r = r[r.series.isin(established(units).columns)
      & (r.target.dt.year == 2025)]
print(f"All 20 series: {coverage(r):.1%} inside, "
      f"{(r.actual < r.q10).mean():.1%} below, "
      f"{(r.actual > r.q90).mean():.1%} above")
for level, name in ((0, "category"), (1, "region")):
    print(f"\nBy {name}          inside   below   above")
    for key, g in r.groupby(r.series.map(lambda s: s[level])):
        print(f"  {key:<16}{coverage(g):>8.1%}"
              f"{(g.actual < g.q10).mean():>8.1%}"
              f"{(g.actual > g.q90).mean():>8.1%}")
