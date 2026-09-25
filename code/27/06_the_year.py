# A year of running Foresight, as a calendar: the lists 2026 holds,
# read from the contracts on record, and for each one the day its
# notices and its outcomes are all in; then what else happens when.
import sqlite3
from contextlib import closing

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import MOMENT_DAYS

with closing(sqlite3.connect(ML_WAREHOUSE)) as con:
    ends = pd.read_sql_query(
        "SELECT DISTINCT end_date FROM contracts", con,
        parse_dates=["end_date"])
ends["mark"] = ends.end_date - pd.Timedelta(days=MOMENT_DAYS)
year = ends[ends.mark.dt.year == 2026].sort_values("mark")
NOTICE = pd.Timedelta(days=60)          # given 60 days before the end
DAY = pd.Timedelta(days=1)              # all on record the next day
year["notices"] = year.end_date - NOTICE + DAY
year["outcomes"] = year.end_date + DAY
year["quarter"] = ~year.mark.dt.quarter.duplicated()

print(f"{'list made':<12}{'notices in':<12}{'outcomes in':<13}"
      "that morning")
for r in year.itertuples():
    extra = "; impact page" if r.quarter else ""
    print(f"{r.mark:%Y-%m-%d}  {r.notices:%Y-%m-%d}  "
          f"{r.outcomes:%Y-%m-%d}   map, gate{extra}")
print(f"\n{len(year)} lists; the last one's outcomes arrive"
      f" {year.outcomes.max():%B %Y}")
print("every night    the job asks whether a list is due")
print("every Monday   the monitoring page, read in five minutes")
print("every quarter  the impact page: arms balanced, nobody held")
print("               out was called; no verdict before its time")
print("every January  the model card and the brief, reread and")
print("               re-signed; the calendar for the next year")
