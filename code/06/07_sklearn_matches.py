# The nine-feature line three ways: our descent, scikit-learn, algebra.
import numpy as np
from sklearn.linear_model import LinearRegression

from foresight.data.spend import (FEATURES, TRAIN, VALIDATION,
                                  baselines, features, scores,
                                  spend_table)
from foresight.models.linear import LinearGD

train = spend_table(*TRAIN)
X, y = features(train), train.spend_next_90d.to_numpy()

ours = LinearGD(lr=0.1, steps=10_000).fit(X, y)
sk = LinearRegression().fit(X, y)
A = np.column_stack([np.ones(len(X)), X])      # a column of ones for b
exact = np.linalg.solve(A.T @ A, A.T @ y)       # the normal equations

print(f"{'weight':<18}{'descent':>15}{'scikit-learn':>15}"
      f"{'algebra':>15}")
rows = zip(["intercept"] + FEATURES,
           np.r_[ours.intercept_, ours.coef_],
           np.r_[sk.intercept_, sk.coef_], exact)
for name, a, b, c in rows:
    print(f"{name:<18}{a:>15.4f}{b:>15.4f}{c:>15.4f}")
gap = np.abs(np.r_[ours.intercept_, ours.coef_]
             - np.r_[sk.intercept_, sk.coef_]).max()
print(f"largest difference, descent vs scikit-learn: {gap:.1e}")

valid = spend_table(*VALIDATION)
v = valid.spend_next_90d.to_numpy()
one = LinearRegression().fit(train[["spend_90d"]], y)
print(f"\nValidation, {len(valid):,} rows")
scores(v, baselines(train, valid) | {
    "a line on spend_90d": one.predict(valid[["spend_90d"]]),
    "nine features": sk.predict(features(valid))})
