# The numbers card, first half: what Foresight found and what it is
# worth, each line read from the record an earlier chapter wrote, each
# with its comparison, and the simulated lines marked as simulated.
import json

from foresight.config import ROOT
from foresight.impact.analyse import dollars
from foresight.impact.report import CALLS_A_YEAR
from foresight.models.logistic import CALLS


def saved(name):
    return json.loads((ROOT / "code" / name).read_text())


year = saved("24/12_champion_challenger.json")
impact = saved("25/11_impact_report.json")
hits, calls = year["hits"], year["calls"]
r, s = impact["retention"], impact["stock"]
a_year = CALLS_A_YEAR                   # forty a list, twelve lists


def k(v):
    return dollars(round(v, -3))


print("Foresight, 2025: what it found and what it is worth\n")
print(f"Leavers in the calls, the {calls // CALLS} lists of 2025"
      f" whose outcomes\nare in ({calls} calls)")
for name in ("as served", "v0.6", "rule", "ceiling"):
    print(f"  {name:<12}{hits[name]:>5}  {hits[name] / calls:>6.1%}")
print(f"  served - rule: {hits['as served'] - hits['rule']} more"
      f" leavers reached, {hits['ceiling'] - hits['as served']}"
      " short of the ceiling")

print("\nRetention calls, SIMULATED at the brief's"
      f" {impact['assumed_save']:.0%} save rate")
print(f"  leaving cut by {100 * r['diff']:.1f} points"
      f" ({100 * r['lo']:.1f} to {100 * r['hi']:.1f})")
v, lo, hi = r["per_call"]
print(f"  a year of {a_year} calls: {k(v * a_year)}"
      f" ({k(lo * a_year)} to {k(hi * a_year)})")
print(f"  the test needs {impact['design']['cohorts']} cohorts;"
      f" a year finds the effect\n  "
      f"{impact['design']['power_one_year']:.0%} of the time")

print("\nStock, measured, under stated costs")
print(f"  {k(s['point'])} a year against last year's month"
      f" ({k(s['low'])} to\n  {k(s['high'])} as the costs move)")
m = impact["medians"]
v08 = m["v0.8's ranking"]
print("\nUrgent tickets, hours to first read (median)")
print(f"  arrival order {m['arrival order']:.1f}, urgent words"
      f" {m['urgent words first']:.1f}, v0.8's ranking {v08:.1f}")
