# Exercise 3: can a ridge penalty rescue the 152-column model?
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from foresight.data.build_table import TABLE
from foresight.evaluate import auc, backtest, hits_at_k
from foresight.models.linear import Standardiser
from foresight.models.logistic import features, log_loss
from foresight.models.regularised import TUNE, maker


class Pairs:
    """Every column and every product of two, then ridge."""

    def __init__(self, strength):
        self.strength = strength

    def grid(self, rows):
        return self.poly_.transform(
            self.scale_.transform(features(rows).to_numpy()))

    def fit(self, rows):
        self.scale_ = Standardiser().fit(features(rows).to_numpy())
        self.poly_ = PolynomialFeatures(2, include_bias=False)
        A = self.poly_.fit_transform(
            self.scale_.transform(features(rows).to_numpy()))
        self.z_ = Standardiser().fit(A)
        C = 1 / (2 * self.strength * len(rows))
        self.m_ = LogisticRegression(C=C, tol=1e-6, max_iter=10_000)
        self.m_.fit(self.z_.transform(A), rows.not_renewed.to_numpy())
        return self

    def predict_proba(self, rows):
        return self.m_.predict_proba(
            self.z_.transform(self.grid(rows)))[:, 1]


table = pd.read_parquet(TABLE)
print(f"{'tuning cohorts':<22}{'log loss':>10}{'AUC':>8}{'top 40':>8}")
runs = [(f"152 columns, {s:g}", lambda s=s: Pairs(s))
        for s in (0.001, 0.003, 0.01, 0.03, 0.1)]
runs.append(("16 columns, lasso", maker("l1", 0.002)))
for name, make in runs:
    b = backtest(table, *TUNE, make)
    y, p = b.not_renewed.to_numpy(), b.model.to_numpy()
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in b.groupby("moment"))
    print(f"{name:<22}{log_loss(y, p):>10.4f}{auc(y, p):>8.4f}"
          f"{hits:>8}")
