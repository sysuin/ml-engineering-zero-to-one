# Chapter 17's demand forecast through 2025, read one month at a time
# as each month's sales arrived: its error against a limit from the
# twelve months before, and how often its 80% intervals held.
# timeout: 300
from foresight.forecast.report import panels
from foresight.forecast.series import PRODUCT_REGION
from foresight.monitor.forecast import by_month, record, watch

P, T = panels(PRODUCT_REGION)
r = record(P, (T,))
months = by_month(r)
w = watch(months, "2025-01")
print(f"One month ahead, {r.series.nunique()} product x region series")
print(f"{'month':<9}{'WAPE':>7}{'limit':>7}{'naive':>7}{'bias':>7}"
      f"{'raw':>7}{'widened':>9}{'recent':>8}")
for m, x in w.iterrows():
    flag = "  alert" if x.error_alert else ""
    print(f"{str(m):<9}{x.wape:>7.1%}{x.limit:>7.1%}{x.naive:>7.1%}"
          f"{x.bias:>+7.1%}{x.raw:>7.1%}{x.widened:>9.1%}"
          f"{x.recent:>8.1%}{flag}")
for year in ("2024", "2025"):
    y = months[months.index.year == int(year)]
    print(f"{year}     {y.wape.mean():>7.1%}{'':>7}"
          f"{y.naive.mean():>7.1%}{y.bias.mean():>+7.1%}"
          f"{y.raw.mean():>7.1%}"
          f"{y.widened.mean():>9.1%}{y.recent.mean():>8.1%}")
print(f"\nCoverage alerts (six months under 60%): "
      f"{int(w.coverage_alert.sum())}")
