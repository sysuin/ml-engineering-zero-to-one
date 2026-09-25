# The daily feed check replayed over three years: every day on which a
# supplier's prices or costs, to one segment of the long tail, moved
# 3% or more from their median over the 56 days before.
from foresight.monitor.feeds import SHIFT, alerts, daily

d = daily()
print(f"{d.supplier.nunique()} series (supplier x segment),"
      f" {d.day.min():%Y-%m-%d} to {d.day.max():%Y-%m-%d}")
found = alerts(d)
print(f"{len(found)} alerts at a shift of {SHIFT:.0%}, on"
      f" {found.day.nunique()} days\n")
print(f"{'day':<12}{'index':<7}{'series':>7}  {'usual':>6}{'today':>7}"
      "  supplier")
for day, g in found.groupby("day"):
    who = sorted(set(g.supplier.str.split(" / ").str[0]))
    name = who[0] if len(who) == 1 else f"{len(who)} suppliers"
    print(f"{day:%Y-%m-%d}  {g.what.iloc[0]:<7}{len(g):>7}"
          f"  {g.usual.median():>6.3f}{g.today.median():>7.3f}  {name}")

print("\nThe Voss alerts of 2025, series by series")
for r in found[found.day.dt.year == 2025].itertuples():
    print(f"  {r.day:%Y-%m-%d} {r.what:<6}{r.supplier:<34}"
          f"{r.today:>6.3f}")

naive = alerts(daily(segment="today"))
near = naive.day.apply(lambda t: (found.day - t).abs().min().days)
extra = naive[near > 7]
print(f"\nWith today's segment instead of the segment on the day:"
      f" {len(naive)} alerts,\n{len(extra)} of them more than a week"
      " from any alert above")
for r in extra.itertuples():
    print(f"  {r.day:%Y-%m-%d} {r.what:<6}{r.supplier:<34}"
          f"{r.today:>6.3f}")
