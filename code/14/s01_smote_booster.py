# Exercise 1: SMOTE on the booster. Ranking and calibration, validation.
import pandas as pd

from foresight.config import SEED, rng
from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, calibration_slope, smote
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.boosting import RenewalBooster


class SmotedBooster(RenewalBooster):
    def fit(self, rows):
        return super().fit(smote(rows, rng(SEED)))


table = pd.read_parquet(TABLE)
runs = {"booster": backtest(table, *SPLITS["validation"],
                            RenewalBooster),
        "with SMOTE": backtest(table, *SPLITS["validation"],
                               SmotedBooster)}
base = by_capacity(runs["booster"])
print(f"{'':<12}{'leavers':>8}{'AUC':>7}{'mean p':>8}{'log loss':>10}"
      f"{'slope':>7}{'shared':>8}")
for name, s in runs.items():
    m = measure(s)
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    print(f"{name:<12}{m['precision']['model'] * 240:>8.0f}"
          f"{m['auc']['model']:>7.3f}{p.mean():>8.1%}"
          f"{m['log loss']['model']:>10.4f}"
          f"{calibration_slope(p, y)[0]:>7.2f}"
          f"{int((by_capacity(s) & base).sum()):>8}")
