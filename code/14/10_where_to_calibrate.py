# Where Platt scaling learns its map: on the rows the model was fitted
# on, on the three latest cohorts held back from it, or (for scale
# only, never in practice) on the validation cohorts it is judged on.
import pandas as pd
from lightgbm import LGBMClassifier

from foresight.config import LIGHTGBM_DETERMINISTIC
from foresight.data.build_table import TABLE
from foresight.decide import Platt, calibrated, calibration_slope
from foresight.evaluate import SPLITS, backtest
from foresight.models.boosting import RenewalBooster, columns
from foresight.models.logistic import log_loss
from foresight.models.regularised import maker


class DefaultBooster:
    """LightGBM as installed, as in the last listing."""

    def fit(self, rows):
        self.model_ = LGBMClassifier(**LIGHTGBM_DETERMINISTIC).fit(
            columns(rows), rows.not_renewed)
        return self

    def predict_proba(self, rows):
        return self.model_.predict_proba(columns(rows))[:, 1]


class InSample:
    """The map learned from the model's scores on its own rows."""

    def __init__(self, make_model):
        self.make_model = make_model

    def fit(self, rows):
        self.model_ = self.make_model().fit(rows)
        self.map_ = Platt().fit(self.model_.predict_proba(rows),
                                rows.not_renewed.to_numpy())
        return self

    def predict_proba(self, rows):
        return self.map_(self.model_.predict_proba(rows))


table = pd.read_parquet(TABLE)
models = {"lasso": maker("l1", 0.002), "booster": RenewalBooster,
          "defaults": DefaultBooster}
print("Validation log loss (calibration slope)")
print(f"{'':<9}{'as fitted':>14}{'own rows':>14}{'held back':>14}"
      f"{'validation':>14}")
for name, make in models.items():
    runs = [backtest(table, *SPLITS["validation"], f)
            for f in (make, lambda: InSample(make), calibrated(make))]
    y = runs[0].not_renewed.to_numpy()
    ps = [s.model.to_numpy() for s in runs]
    ps.append(Platt().fit(ps[0], y)(ps[0]))    # the cheat, for scale
    print(f"{name:<9}" + "".join(
        f"{log_loss(y, p):>7.4f} ({calibration_slope(p, y)[0]:.2f})"
        for p in ps))
