# v0.4's lasso fitted three ways: on the contracts as they are, with
# leavers copied until they match renewers, and with SMOTE's invented
# leavers. Same validation backtest, same cohorts.
import pandas as pd

from foresight.config import SEED, rng
from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, oversample, smote
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.regularised import RegularisedRisk, compare


class Resampled:
    """The lasso, fitted on resampled rows; scored on real ones."""

    def __init__(self, how=None):
        self.how = how

    def fit(self, rows):
        if self.how is not None:
            rows = self.how(rows, rng(SEED))
        self.model_ = RegularisedRisk("l1", 0.002).fit(rows)
        return self

    def predict_proba(self, rows):
        return self.model_.predict_proba(rows)


table = pd.read_parquet(TABLE)
runs = {name: backtest(table, *SPLITS["validation"],
                       lambda how=how: Resampled(how))
        for name, how in (("as fitted", None),
                          ("oversampled", oversample),
                          ("SMOTE", smote))}
plain = runs["as fitted"]
print(f"Validation: {len(plain):,} contracts,"
      f" {plain.not_renewed.mean():.1%} left\n")
print(f"{'':<13}{'leavers':>8}{'AUC':>7}{'mean p':>8}{'log loss':>10}"
      f"{'Brier':>8}{'shared':>8}")
base = by_capacity(plain)
for name, s in runs.items():
    m = measure(s)
    shared = int((by_capacity(s) & base).sum())
    print(f"{name:<13}{m['precision']['model'] * 240:>8.0f}"
          f"{m['auc']['model']:>7.3f}{s.model.mean():>8.1%}"
          f"{m['log loss']['model']:>10.4f}{m['brier']['model']:>8.4f}"
          f"{shared:>8}")
print("shared: calls in common with the lasso as fitted, of 240")

print("\nMinus the lasso as fitted, paired (95%)")
for name in ("oversampled", "SMOTE"):
    d, lo, hi = compare(runs[name], plain)["precision"]
    print(f"  {name:<12} precision at 40 {d * 100:+.1f} points"
          f" ({lo * 100:+.1f} to {hi * 100:+.1f})")
