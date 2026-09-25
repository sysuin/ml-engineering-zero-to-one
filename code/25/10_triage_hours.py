# How long 2025's Urgent tickets would have waited for a first read,
# with the queue read in arrival order, by urgent words, and by v0.8.
import json

import numpy as np
import pandas as pd

from foresight.impact.queue import Desk, first_read
from foresight.triage.evaluate import keyword_rule
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel

t = tickets()
t["category"] = relabel(t)
train, valid, test = split(t)
seen = pd.concat([train, valid], ignore_index=True)
model = TriageModel().fit(seen.body, seen.priority, seen.category)
scores = {"arrival order": np.zeros(len(test)),
          "urgent words first": test.body.map(keyword_rule).to_numpy(),
          "v0.8's P(Urgent)": model.predict(test.body).p_urgent
          .to_numpy()}
urgent = (test.priority == "Urgent").to_numpy()

desk = Desk()
print(f"Assumed: the desks read from {desk.opens:.0f}:00 to"
      f" {desk.closes:.0f}:00 every day,")
print(f"one ticket every {desk.minutes:.0f} minutes.")
print(f"2025: {len(test):,} tickets, {urgent.sum()} Urgent"
      " (the desks' label)\n")
print(f"{'read in order of':<20}{'Urgent: median':>15}{'90th pct':>10}"
      f"{'in 1 hour':>11}{'all: mean':>11}")
out = {}
for name, s in scores.items():
    wait = first_read(test.opened_at, s, desk)
    u = wait[urgent]
    out[name] = np.sort(u).tolist()
    print(f"{name:<20}{np.median(u):>13.1f} h"
          f"{np.quantile(u, 0.9):>8.1f} h{(u <= 1).mean():>11.0%}"
          f"{wait.mean():>9.1f} h")

print("\nUrgent median wait, hours, as the desk's pace moves")
print(f"{'minutes a ticket':<20}" + "".join(
    f"{m:>8}" for m in (30, 40, 45)))
for name, s in scores.items():
    cells = [np.median(first_read(test.opened_at, s, Desk(minutes=m))
                       [urgent]) for m in (30, 40, 45)]
    print(f"{name:<20}" + "".join(f"{c:>8.1f}" for c in cells))
with open("code/25/10_triage_hours.json", "w") as f:
    json.dump(out, f)
