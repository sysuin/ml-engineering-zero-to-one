# Foresight's first months live, as the record shows them: the order
# feed recomputed here, and the lists, notices and alerts read from
# what Chapter 24's listings wrote. Nothing in it is a story.
import json
import re

import pandas as pd

from foresight.config import ROOT
from foresight.monitor import feeds

WAIT = pd.Timedelta(days=31)        # a list's notices are all in
drift = (ROOT / "code/24/07_score_drift.out").read_text()
(rows,) = re.search(r"its ([\d,]+)\s+training rows", drift).groups()
log = [(pd.Timestamp("2025-01-01"), f"go-live: v0.6, fitted on the"
        f" {rows} outcomes known that morning")]

fed = feeds.alerts(feeds.daily(), "2025-01-01", "2025-12-31")
fed["change"] = fed.today / fed.usual - 1
for (day, what), g in fed.groupby(["day", "what"]):
    who = g.supplier.str.split(" / ").str[0].iloc[0]
    log.append((day, f"order feed: {who} {what} up"
                f" {g.change.median():.0%} on {len(g)} series: alert"))

for mark, n, mean, psi in re.findall(
        r"(2025-0[1-3]-\d\d)\s+(\d+)\s+\d+\s+(\S+%)\s+(\S+)", drift):
    log.append((pd.Timestamp(mark), f"list: {n} contracts, chances"
                f" average {mean}, PSI {psi}"))

notices = json.loads((ROOT / "code/24/04_notices.json").read_text())
for mark in ("2025-01-30", "2025-03-02"):
    n, m = notices[mark], pd.Timestamp(mark)
    loud = (": alert" if not n["low"] <= n["noticed"] <= n["high"]
            else ": quiet")
    log.append((m + WAIT, f"the {m.day} {m:%B} list's"
                f" notices: {n['noticed']:.0f} against"
                f" {n['expected']:.1f} expected ({n['low']:.0f} to"
                f" {n['high']:.0f}){loud}"))

page = json.loads((ROOT / "code/24/13_monitoring_report.json")
                  .read_text())["2025-12-31"]
got, expected, low, high = re.search(
    r"2025-10-02\s+(\d+)\s+(\S+)\s+(\d+) to (\d+)", page).groups()
log.append((pd.Timestamp("2025-10-02") + WAIT, f"the new champion's"
            f" first notices: {got} against {expected} ({low} to"
            f" {high}): alert"))

for day, said in sorted(log, key=lambda x: x[0]):
    words, line, first = said.split(), "", True
    for w in words:
        if len(line) + len(w) + 1 > 54:
            print(f"{day:%Y-%m-%d}  {line}" if first
                  else f"{'':12}{line}")
            line, first = "", False
        line = f"{line} {w}".strip()
    print(f"{day:%Y-%m-%d}  {line}" if first else f"{'':12}{line}")
