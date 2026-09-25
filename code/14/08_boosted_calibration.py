# LightGBM with its default settings beside Foresight's booster and the
# lasso: how bold each one's chances are, on validation and on the rows
# it was fitted on.
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.data.build_table import TABLE
from foresight.decide import calibration_slope, reliability
from foresight.evaluate import (HISTORY_FROM, SPLITS, backtest,
                                known_by, measure)
from foresight.models.boosting import RenewalBooster, columns
from foresight.models.logistic import log_loss
from foresight.models.regularised import maker


class DefaultBooster:
    """LightGBM as installed: 100 trees of 31 leaves, no stopping."""

    def fit(self, rows):
        self.model_ = LGBMClassifier(**LIGHTGBM_DETERMINISTIC).fit(
            columns(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows):
        return self.model_.predict_proba(columns(rows))[:, 1]


table = pd.read_parquet(TABLE)
runs = {"lasso": maker("l1", 0.002), "booster": RenewalBooster,
        "defaults": DefaultBooster}
print(f"{'validation':<11}{'leavers':>8}{'AUC':>7}{'log loss':>10}"
      f"{'slope':>7}{'top bin said':>14}{'left':>7}")
for name, make in runs.items():
    s = backtest(table, *SPLITS["validation"], make)
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    m, top = measure(s), reliability(p, y).iloc[-1]
    print(f"{name:<11}{m['precision']['model'] * 240:>8.0f}"
          f"{m['auc']['model']:>7.3f}{m['log loss']['model']:>10.4f}"
          f"{calibration_slope(p, y)[0]:>7.2f}{top.predicted:>14.1%}"
          f"{top.left:>7.1%}")

first = table[table.end_date >= SPLITS["validation"][0]].moment.min()
rows = known_by(table[table.end_date >= HISTORY_FROM], first)
print(f"\nOn the {len(rows):,} rows each was fitted on at the"
      f" {first.day} {first:%B} mark")
print(f"{'':<11}{'log loss':>10}{'slope':>7}")
for name, make in runs.items():
    p = make().fit(rows).predict_proba(rows)
    y = rows.not_renewed.to_numpy()
    print(f"{name:<11}{log_loss(y, p):>10.4f}"
          f"{calibration_slope(p, y)[0]:>7.2f}")
