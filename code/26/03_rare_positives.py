# Rare positives: how many you need, and what downsampling does.
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from foresight.config import SEED, rng

# 1. How precisely can a validation set measure recall of 60%?
z = norm.ppf(0.975)
print("Recall of 60%, measured on n frauds: 95% interval")
for n in (50, 200, 1_000, 5_000):
    half = z * np.sqrt(0.6 * 0.4 / n)
    print(f"  n = {n:>5,}   60% +/- {half:.1%}")
need = z**2 * 0.6 * 0.4 / 0.02**2
print(f"  frauds needed for +/- 2 points: {need:,.0f}")

# 2. A synthetic payments table: three signals, about 1 fraud in 500.
g = rng()
N = 1_000_000
X = g.normal(size=(N, 3))
logit = -7.3 + 1.2 * X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 2]
y = g.random(N) < 1 / (1 + np.exp(-logit))
train, test = np.arange(N) < N // 2, np.arange(N) >= N // 2
print(f"\n{N:,} payments, {y.mean():.2%} fraud")

# Keep every fraud and 2% of the legitimate payments.
keep = 0.02
sample = train & (y | (g.random(N) < keep))
print(f"Training rows: {train.sum():,} in full,"
      f" {sample.sum():,} downsampled")

fit = dict(C=1.0, max_iter=1000, random_state=SEED)
full = LogisticRegression(**fit).fit(X[train], y[train])
down = LogisticRegression(**fit).fit(X[sample], y[sample])


def corrected(q):
    """Undo the sampling: the odds were inflated by 1 / keep."""
    return keep * q / (keep * q + 1 - q)


p_full = full.predict_proba(X[test])[:, 1]
p_down = down.predict_proba(X[test])[:, 1]
actual = y[test].mean()
print(f"\n{'on the test half':24}{'AUC':>7}{'mean p':>9}"
      f"{'vs actual':>11}")
for name, p in (("all rows", p_full), ("downsampled", p_down),
                ("downsampled, corrected", corrected(p_down))):
    print(f"{name:24}{roc_auc_score(y[test], p):>7.3f}"
          f"{p.mean():>9.3%}{p.mean() / actual:>10.2f}x")
