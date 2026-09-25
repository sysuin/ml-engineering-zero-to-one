# Exercise 1: descent on absolute error instead of squared error.
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor

from foresight.data.spend import (TRAIN, VALIDATION, baselines,
                                  features, scores, spend_table,
                                  year_on_record)
from foresight.models.linear import Standardiser

everything = spend_table(*TRAIN)
train = everything[year_on_record(everything)]
valid = spend_table(*VALIDATION)
X, y = features(train), train.spend_next_90d.to_numpy()
scaler = Standardiser().fit(X)
Z, Zv = scaler.transform(X), scaler.transform(features(valid))


def absolute_error_gradient(X, y, w, b):
    """Only the sign of each miss counts, not its size."""
    sign = np.sign(X @ w + b - y)
    return X.T @ sign / len(y), sign.mean()


w, b = np.zeros(Z.shape[1]), 0.0
for lr in (100.0, 10.0, 1.0):              # smaller steps as it settles
    for _ in range(20_000):
        dw, db = absolute_error_gradient(Z, y, w, b)
        w, b = w - lr * dw, b - lr * db

squared = LinearRegression().fit(Z, y)
v = valid.spend_next_90d
print(f"Validation, {len(valid):,} rows; both fitted on "
      f"{len(train):,} full-year rows")
scores(v, baselines(everything, valid) | {
    "squared error": squared.predict(Zv),
    "absolute error": Zv @ w + b})
exact = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
exact.fit(Z, y)
print(f"  {'absolute, solved exactly':<34}"
      f"{np.mean(abs(v - exact.predict(Zv))):>10,.0f}")
print("\ntraining rows predicted above what they spent:")
print(f"  absolute {np.mean(Z @ w + b > y):.1%}"
      f"   squared {np.mean(squared.predict(Z) > y):.1%}")
print(f"{'weight per std dev':<18}{'squared':>10}{'absolute':>10}")
for name, a, c in zip(("spend_90d", "spend_365"), squared.coef_, w):
    print(f"{name:<18}{a:>10,.0f}{c:>10,.0f}")
print(f"{'intercept':<18}{squared.intercept_:>10,.0f}{b:>10,.0f}")
