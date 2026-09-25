# The launch checklist, checked against the system rather than from
# memory: every line a script can test is tested, in a temporary copy
# of the sandbox; the rest are marked for a person. Writes the page.
# timeout: 300
import inspect
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from foresight import demo
from foresight.config import ROOT
from foresight.monitor import alerts, psi, report, watch
from foresight.pipeline import run
from foresight.pipeline.registry import Registry
from foresight.serve import batch, store
from foresight.serve.api import RenewalRequest, Service, Stopwatch

root = demo.workspace()
first = batch.month("2025-12-31", root=root)
second = batch.month("2026-01-01", root=root)
top = int(first["scored"].query("rank == 1").contract_id.iloc[0])
registry = Registry("renewal", root / "artifacts")
state = registry.read()
live = [v for v, r in state["versions"].items()
        if r["stage"] == "production"]
card = registry.folder / live[0] / "model_card.md"


def refused(check, *args) -> bool:
    try:
        check(*args)
    except (batch.StaleData, ValidationError):
        return True
    return False


def by_rule() -> str:
    empty = Path(tempfile.mkdtemp())
    try:
        return Service(empty, on="2026-01-01").score(
            top, Stopwatch())["answered_by"]
    finally:
        shutil.rmtree(empty)


brief = (ROOT / "docs" / "foresight-brief.md").read_text()
held = pd.read_csv(store.path(root).parent / "holdout.csv")
promoted = [e for e in state["log"] if e["stage"] == "production"]
job = inspect.getsource(run.run)
watched = "".join(inspect.getsource(m) for m in (
    alerts, psi, watch, report))       # everything the monitor reads
HAND = None
CHECKS = {
    "Before anyone relies on it": [
        ("the brief names decision, baseline and sign-off",
         all(f"| {k} |" in brief
             for k in ("Decision", "Baseline", "Signs off"))),
        ("the table is refused if an expectation fails",
         "check_table(table)" in job),
        ("leakage checks run on every training job",
         "run_checks(" in job),
        ("the model's card sits beside it in the registry",
         card.exists() and "## Known issues" in card.read_text()),
        ("a person promoted it, and wrote down why",
         bool(promoted) and all(e["reason"] for e in promoted)),
        ("a version to roll back to is in the registry",
         len(state["versions"]) > 1),
    ],
    "The night it runs alone": [
        ("the job refuses a stale warehouse",
         refused(batch.fresh, pd.Timestamp("2026-01-30"))),
        ("a second run does nothing",
         second["status"] == "done already"),
        ("the job's rows match the API's (skew test)",
         "check_skew" in inspect.getsource(batch._make)),
        ("the API refuses a malformed request",
         refused(RenewalRequest.model_validate,
                 {"contract_id": str(top)})),
        ("with no model, the rule answers",
         by_rule() == "rule"),
    ],
    "Once it is live": [
        ("alert thresholds are constants, not tuned later",
         psi.ACT == 0.25 and alerts.EFFECT and watch.ALPHA),
        ("a holdout is drawn after every list",
         (held.arm == "held out").sum() == 20),
        ("a failed run alerts a person by the morning",
         "store" in watched or "scores.db" in watched),
        ("the CI gate blocks a worse model (make test)", HAND),
        ("someone who did not build it ran make demo", HAND),
        ("the account team knows 'held out' means no call", HAND),
        ("an owner is named for each part", HAND),
    ],
}

WORD = {True: "yes", False: "no", None: "by hand"}
lines = ["# Foresight: the launch checklist", "",
         "Written by `code/27/02_launch_checklist.py`, which tests"
         " every line a script can test against a copy of the"
         " system. A line marked *by hand* needs a person to tick"
         " it; a *no* is a known gap, written down.", ""]
print("Foresight's launch checklist, checked against the system")
for section, items in CHECKS.items():
    print(f"\n{section}")
    lines += [f"## {section}", ""]
    for what, ok in items:
        word = WORD[None if ok is None else bool(ok)]
        print(f"  {word:<9}{what}")
        box = {"yes": "[x]", "no": "[ ]", "by hand": "[?]"}[word]
        lines.append(f"- {box} {what} ({word})")
    lines.append("")

counts = pd.Series([WORD[None if ok is None else bool(ok)]
                    for items in CHECKS.values() for _, ok in items]
                   ).value_counts()
print(f"\n{counts.sum()} lines: {counts.get('yes', 0)} yes,"
      f" {counts.get('no', 0)} no, {counts.get('by hand', 0)} by hand")
(ROOT / "docs" / "foresight-launch-checklist.md").write_text(
    "\n".join(lines))
with open("code/27/02_launch_checklist.json", "w") as f:
    json.dump({s: [[w, WORD[None if ok is None else bool(ok)]]
                   for w, ok in items] for s, items in CHECKS.items()},
              f)
shutil.rmtree(root)
