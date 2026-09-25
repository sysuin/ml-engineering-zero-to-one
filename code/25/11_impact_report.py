# The one-page impact report for finance, filled from this chapter's
# listings: the retention test (simulated), stock, and tickets.
import json

import numpy as np

from foresight.impact.analyse import cohorts_needed, power
from foresight.impact.queue import Desk
from foresight.impact.report import write


def numbers(name):
    with open(f"code/25/{name}.json") as f:
        return json.load(f)


test, stock = numbers("06_the_test"), numbers("09_forecast_value")
waits, design = numbers("10_triage_hours"), numbers("04_power")
p40, save = design["p40"], 0.25
cohorts = int(np.ceil(cohorts_needed(p40, p40 * (1 - save), 20, 20)))
moves = [row[2] for row in stock["sensitivity"]]
desk = Desk()
findings = {
    "year": 2025, "simulated": True, "assumed_save": save,
    "retention": test,
    "design": {"cohorts": cohorts,
               "power_one_year": power(p40, p40 * (1 - save), 240)},
    "stock": {"series": 190,
              "point": stock["seasonal naive"]["total"]
              - stock["model, median"]["total"],
              "range": stock["seasonal naive x 1.20"]["total"]
              - stock["model, 90th percentile"]["total"],
              "low": min(moves), "high": max(moves)},
    "triage": {"n": len(waits["arrival order"]),
               "waits": {"arrival order": waits["arrival order"],
                         "urgent words first":
                             waits["urgent words first"],
                         "v0.8's ranking": waits["v0.8's P(Urgent)"]},
               "desk": f"{desk.opens:.0f}:00 to {desk.closes:.0f}:00,"
                       f" a ticket every {desk.minutes:.0f} minutes"},
    "ask": [f"Keep 20 of each list uncalled until cohort {cohorts}"
            " (September", "2027), read the test once a year, and"
            " judge the calls on the", "final reading. Until then,"
            " cost the calls at the brief's 25%."]}
print(write(findings))
with open("code/25/11_impact_report.json", "w") as f:
    json.dump({k: v for k, v in findings.items() if k != "triage"}
              | {"medians": {k: float(np.median(v)) for k, v in
                             findings["triage"]["waits"].items()}}, f)
