# Exercise 2: the model against itself without regions, power again.
import numpy as np
import pandas as pd
from scipy.stats import norm

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, backtest
from foresight.models.logistic import RenewalRisk

table = pd.read_parquet(TABLE)


class NoRegion(RenewalRisk):
    """The same model with every contract placed in the Southwest,
    the reference region: the region columns carry nothing."""

    def fit(self, rows):
        return super().fit(rows.assign(region="Southwest"))

    def predict_proba(self, rows):
        return super().predict_proba(rows.assign(region="Southwest"))


full = backtest(table, *SPLITS["validation"])
lean = backtest(table, *SPLITS["validation"], NoRegion)
y = full.not_renewed.to_numpy()
a, b = full.model.to_numpy(), lean.model.to_numpy()
r = full.rule.to_numpy()
print(f"AUC: model {auc(y, a):.4f}, without regions {auc(y, b):.4f},"
      f" rule {auc(y, r):.4f}")
cells = [np.flatnonzero(full.moment.to_numpy() == m)
         for m in full.moment.unique()]
g = rng()
d_near, d_rule = [], []
for _ in range(1000):
    ix = np.concatenate([c[g.integers(0, len(c), len(c))]
                         for c in cells])
    d_near.append(auc(y[ix], a[ix]) - auc(y[ix], b[ix]))
    d_rule.append(auc(y[ix], a[ix]) - auc(y[ix], r[ix]))
z = norm.ppf(0.975) + norm.ppf(0.80)
print(f"{'pair':<30}{'SD of AUC gap':>14}{'months for 0.02':>17}")
for name, d in (("model v. no regions", d_near),
                ("model v. rule", d_rule)):
    s = np.std(d)
    print(f"  {name:<28}{s:>14.4f}{6 * (z * s / 0.02) ** 2:>17.1f}")
