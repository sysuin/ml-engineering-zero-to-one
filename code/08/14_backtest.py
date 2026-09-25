# Foresight v0.3's evaluation on the validation cohorts, as one page.
from foresight.evaluate import REPORT, evaluate, report

ev = evaluate("validation")
print(report(ev, REPORT))
