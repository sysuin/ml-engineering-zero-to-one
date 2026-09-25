# What the monitor knows about each 2025 list on the morning of
# 1 July 2025: every list has inputs and scores, fewer have notices,
# fewer still have outcomes.
import json

import pandas as pd

from foresight.monitor.cohorts import as_scored, notice_due, noticed

rows = as_scored("2025-01-01", "2026-03-31")
rows = rows[rows.moment >= "2025-01-01"]
DAY = pd.Timestamp("2025-07-01")

print(f"On {DAY:%d %B %Y}, list by list")
print(f"{'list':<12}{'contracts':>10}{'notices by':>12}{'notices':>9}"
      f"{'ends':>9}{'left':>7}")
for mark, c in rows[rows.moment <= DAY].groupby("moment"):
    due, end = notice_due(c).max(), c.end_date.max()
    seen = f"{int(noticed(c, DAY).sum())}" if due < DAY else "-"
    left = f"{int(c.not_renewed.sum())}" if end < DAY else "-"
    by, ends = f"{due:%d %b}", f"{end:%d %b}"
    print(f"{mark:%Y-%m-%d}{len(c):>12}{by:>12}{seen:>9}{ends:>9}"
          f"{left:>7}")

print("\nDays from the mark to each kind of evidence, 2025's lists")
days = {"notices due": (notice_due(rows) - rows.moment).dt.days,
        "outcome on record": (rows.end_date + pd.Timedelta(days=1)
                              - rows.moment).dt.days}
for k, v in days.items():
    lo, hi = v.min(), v.max()
    span = f"{lo}" if lo == hi else f"{lo}-{hi}"
    print(f"  {k:<20}{span:>4} days")
march = rows[rows.moment == "2025-03-02"]
with open("code/24/02_label_delay.json", "w") as f:
    json.dump({"mark": "2025-03-02",
               "notice": f"{notice_due(march).max():%Y-%m-%d}",
               "end": f"{march.end_date.max():%Y-%m-%d}"}, f, indent=1)
