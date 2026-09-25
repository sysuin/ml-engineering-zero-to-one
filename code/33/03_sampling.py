# How far a rate moves by chance: the formula, and the bootstrap.
import numpy as np

from foresight.config import rng
from foresight.models.logistic import TRAIN, load

train = load(*TRAIN)
y = train.not_renewed.to_numpy()
legacy = train.legacy_terms.to_numpy() == 1
p = y.mean()

# Draw samples of n contracts from the training rows, many times,
# and watch the spread of the leave rate shrink as n grows.
g = rng()
print(f"Leave rate {p:.4f}. Samples drawn 2,000 times at each size:")
print(f"{'n':>7}{'spread seen':>13}{'formula':>10}")
for n in (250, 1_000, 4_000):
    rates = y[g.integers(0, len(y), (2_000, n))].mean(axis=1)
    formula = np.sqrt(p * (1 - p) / n)
    print(f"{n:>7,}{rates.std():>13.4f}{formula:>10.4f}")


def normal_interval(k, n):
    """The rate k/n, plus or minus 1.96 standard errors."""
    r = k / n
    half = 1.96 * np.sqrt(r * (1 - r) / n)
    return r, r - half, r + half


print("\n95% intervals by formula")
for name, rows in (("legacy", legacy), ("current", ~legacy)):
    r, lo, hi = normal_interval(y[rows].sum(), rows.sum())
    print(f"  {name:<8}{r:.4f}   {lo:.4f} to {hi:.4f}")

# The bootstrap: resample, recompute, repeat. 1,230 accounts have
# two contracts here, so the unit drawn is the account.
accounts = train.groupby("account_id").indices    # rows per account
units = list(accounts.values())
n_units = len(units)
d = y[legacy].mean() - y[~legacy].mean()
print(f"\nLegacy minus current {d:+.4f}; 95% bootstrap intervals:")
for name, draw in (
        ("contracts", lambda: g.integers(0, len(y), len(y))),
        ("accounts", lambda: np.concatenate(
            [units[k] for k in g.integers(0, n_units, n_units)]))):
    rates, diffs = [], []
    for _ in range(2_000):
        i = draw()
        yb, lb = y[i], legacy[i]
        rates.append(yb[lb].mean())
        diffs.append(yb[lb].mean() - yb[~lb].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    ahead = np.mean(np.array(diffs) > 0)
    print(f"  drawing {name:<10} {lo:+.4f} to {hi:+.4f},"
          f" legacy ahead in {ahead:.1%}")
lo, hi = np.percentile(rates, [2.5, 97.5])
print(f"Legacy rate alone, by account: {lo:.4f} to {hi:.4f}")
