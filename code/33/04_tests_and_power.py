# Is a difference real? A test, a shuffle, pairing, and sample size.
import numpy as np
from scipy.stats import norm

from foresight.config import rng
from foresight.models.logistic import (TRAIN, at_capacity,
                                       days_since_rule, load)

train = load(*TRAIN)
y = train.not_renewed.to_numpy()
legacy = train.legacy_terms.to_numpy() == 1

# 1. Two proportions: legacy against current terms.
n1, n0 = legacy.sum(), (~legacy).sum()
p1, p0 = y[legacy].mean(), y[~legacy].mean()
pooled = y.mean()
se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
z = (p1 - p0) / se
print(f"Legacy {p1:.4f} against current {p0:.4f}:"
      f" z = {z:.2f}, p-value {2 * norm.sf(abs(z)):.4f}")

# 2. The same question by shuffling the labels.
g = rng()
observed = p1 - p0
shuffled = []
for _ in range(2_000):
    s = g.permutation(legacy)
    shuffled.append(y[s].mean() - y[~s].mean())
extreme = np.mean(np.abs(shuffled) >= abs(observed))
print(f"Shuffled 2,000 times: {extreme:.4f} of shuffles differ"
      " by as much")

# 3. Pairing: orders in the 90 days before the mark against the
#    90 days before that, for the same contracts.
now = train.orders_90d.to_numpy(float)
before = train.orders_prev_90d.to_numpy(float)
diff = now - before
n = len(diff)
paired = diff.std(ddof=1) / np.sqrt(n)
unpaired = np.sqrt(now.var(ddof=1) / n + before.var(ddof=1) / n)
print(f"\nOrders, last 90 days minus the 90 before: mean"
      f" {diff.mean():+.3f}")
print(f"  standard error paired {paired:.3f},"
      f" as if unpaired {unpaired:.3f}")
print(f"  correlation of the two quarters"
      f" {np.corrcoef(now, before)[0, 1]:.3f}")

# 4. Sizing a holdout. Of the rule's 40 calls a cohort, how many
#    would have left? If a call saves one in four, how many
#    contracts per group does a test need to see it?
base = at_capacity(train, days_since_rule(train))["precision"]
treated = base * (1 - 0.25)
za, zb = norm.ppf(0.975), norm.ppf(0.80)
spread = base * (1 - base) + treated * (1 - treated)
per_group = (za + zb) ** 2 * spread / (base - treated) ** 2
print(f"\nHeld out {base:.3f} would leave; called {treated:.3f}")
print(f"80% power at 5%: {per_group:,.0f} contracts per group")
for size in (20, 40):
    print(f"  at {size} per group a cohort:"
          f" {np.ceil(per_group / size):.0f} monthly cohorts")
print(f"{'per group':>10}{'power':>8}")
for m in (250, 500, 1_000, 2_000):
    s = np.sqrt(spread / m)
    print(f"{m:>10,}{norm.cdf((base - treated) / s - za):>8.0%}")
