# How long each design runs, when a cohort's answer arrives as a
# notice a month after its calls and as an outcome a quarter after.
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

with sqlite3.connect(ML_WAREHOUSE) as con:
    c = pd.read_sql_query("""
        SELECT end_date, outcome, notice_date FROM contracts
        WHERE end_date BETWEEN '2025-01-01' AND '2025-12-31'""", con,
        parse_dates=["end_date", "notice_date"])
left = c[c.outcome == "not_renewed"]
ahead = (left.end_date - left.notice_date).dt.days
print(f"2025 leavers: {len(left)}; with a notice {ahead.notna().sum()},"
      f" all {ahead.min():.0f} days before the end")
print(f"stayers with a notice: "
      f"{c[c.outcome == 'renewed'].notice_date.notna().sum()}\n")

first = pd.Period("2025-01", freq="M")     # the first 2025 cohort's end
designs = {}
print(f"{'design':<22}{'cohorts':>8}{'last list':>12}"
      f"{'last notice':>13}{'last outcome':>13}")
for name, cohorts in [("one year, any design", 12),
                      ("top 80, 40 held out", 26),
                      ("top 40, 20 held out", 35),
                      ("top 40, 10 held out", 48)]:
    end = (first + cohorts - 1).end_time.normalize()
    mark, notice = (end - pd.Timedelta(days=d) for d in (90, 60))
    dates = [f"{d:%b %Y}" for d in (mark, notice, end)]
    designs[name] = [cohorts, *(f"{d:%Y-%m-%d}" for d in (mark, notice,
                                                         end))]
    print(f"{name:<22}{cohorts:>8}{dates[0]:>12}{dates[1]:>13}"
          f"{dates[2]:>13}")
with open("code/25/05_how_long.json", "w") as f:
    json.dump(designs, f)
