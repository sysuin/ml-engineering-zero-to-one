# The milestone's one look at the test year: v0.3 on 2025's cohorts.
from foresight.config import ROOT
from foresight.evaluate import evaluate, report

ev = evaluate("test")
print(report(ev, ROOT / "docs" / "foresight-v0.3-test.md",
             "Foresight v0.3 on the test year: reported once"))
