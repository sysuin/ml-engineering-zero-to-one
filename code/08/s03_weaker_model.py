# Exercise 3: the one-page report for a model without the rule's column.
import tempfile
from pathlib import Path

from foresight.evaluate import evaluate, report
from foresight.models.logistic import RenewalRisk


class NoRecency(RenewalRisk):
    """Days since the last order set to a year for every contract, so
    the column carries nothing and the model must do without it."""

    def fit(self, rows):
        return super().fit(rows.assign(days_since_order=365))

    def predict_proba(self, rows):
        return super().predict_proba(rows.assign(days_since_order=365))


ev = evaluate("validation", make_model=NoRecency)
with tempfile.TemporaryDirectory() as tmp:
    page = report(ev, Path(tmp) / "page.md",
                  "Foresight evaluation: without recency")
print("\n".join(page.splitlines()[:27]))
