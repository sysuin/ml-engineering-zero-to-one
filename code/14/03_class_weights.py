# Class weights on v0.4's lasso and on the booster: each leaver counts
# as many times as there are renewers per leaver. Then the lasso's
# weighting undone, by dividing the odds by the same number.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.decide import undo_weight
from foresight.evaluate import (HISTORY_FROM, SPLITS, backtest,
                                known_by, measure)
from foresight.models.boosting import RenewalBooster
from foresight.models.linear import Standardiser
from foresight.models.regularised import RegularisedRisk


class WeightedLasso(RegularisedRisk):
    def __init__(self, undo=False):
        super().__init__("l1", 0.002)
        self.undo = undo

    def fit(self, rows):
        X = self.columns(rows)
        self.columns_ = list(X.columns)
        self.scaler_ = Standardiser().fit(X.to_numpy())
        y = rows.not_renewed.to_numpy()
        self.ratio_ = (y == 0).sum() / (y == 1).sum()
        weight = np.where(y == 1, self.ratio_, 1.0)
        m = LogisticRegression(l1_ratio=1, solver="liblinear",
                               C=1 / (self.strength * weight.sum()),
                               tol=1e-8, intercept_scaling=100,
                               max_iter=10_000, random_state=SEED)
        self.model_ = m.fit(self.scaler_.transform(X.to_numpy()), y,
                            sample_weight=weight)
        return self

    def predict_proba(self, rows):
        p = super().predict_proba(rows)
        return undo_weight(p, self.ratio_) if self.undo else p


class WeightedBooster(RenewalBooster):
    def fit(self, rows):
        y = rows.not_renewed.to_numpy()
        self.ratio_ = (y == 0).sum() / (y == 1).sum()
        return super().fit(rows)

    def params(self, rounds):
        return {**super().params(rounds),
                "scale_pos_weight": self.ratio_}


table = pd.read_parquet(TABLE)
models = {"lasso": lambda: RegularisedRisk("l1", 0.002),
          "lasso, weighted": WeightedLasso,
          "  then undone": lambda: WeightedLasso(undo=True),
          "booster": RenewalBooster,
          "booster, weighted": WeightedBooster}
print(f"{'validation':<19}{'leavers':>8}{'AUC':>7}{'mean p':>8}"
      f"{'log loss':>10}{'Brier':>8}")
for name, make in models.items():
    s = backtest(table, *SPLITS["validation"], make)
    m = measure(s)
    print(f"{name:<19}{m['precision']['model'] * 240:>8.0f}"
          f"{m['auc']['model']:>7.3f}{s.model.mean():>8.1%}"
          f"{m['log loss']['model']:>10.4f}{m['brier']['model']:>8.4f}")

first = table[table.end_date >= SPLITS["validation"][0]].moment.min()
rows = known_by(table[table.end_date >= HISTORY_FROM], first)
print(f"\nThe booster fitted at the {first.day} {first:%B %Y} mark")
for name, make in (("unweighted", RenewalBooster),
                   ("weighted", WeightedBooster)):
    b = make().fit(rows)
    print(f"  {name:<11} stopped at round {b.rounds_:>4,};"
          f" watched log loss {min(b.watched_):.4f}")
