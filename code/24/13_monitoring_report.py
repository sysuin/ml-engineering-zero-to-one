# The monitoring report: the year lived once, then read as of two
# mornings. Writes docs/foresight-monitoring-2025.md and prints the
# last page of the year.
# timeout: 400
import json

from foresight.monitor.report import REPORT, gather, page, write

year = gather()
days = ["2025-04-07", "2025-12-31"]
write(year, days)
print(page(year, days[-1]))
print(f"\n{len(year.alerts)} alerts in the year; written to"
      f" {REPORT.name}")
with open("code/24/13_monitoring_report.json", "w") as f:
    json.dump({d: page(year, d) for d in days}, f, indent=1)
