# Centre and spread of two training columns, by hand and by pandas.
import numpy as np

from foresight.models.logistic import TRAIN, load

train = load(*TRAIN)
gap = train.days_since_order.dropna().astype(float).to_numpy()
n = len(gap)
mean = gap.sum() / n
variance = ((gap - mean) ** 2).sum() / (n - 1)
sd = variance ** 0.5
print(f"Days since the last order, {n:,} contracts")
print(f"  mean {mean:.2f}, by pandas"
      f" {train.days_since_order.mean():.2f}")
print(f"  variance {variance:,.1f} square days,"
      f" standard deviation {sd:.2f} days")
q1, median, q3 = np.percentile(gap, [25, 50, 75])
print(f"  quartiles {q1:.0f}, {median:.0f}, {q3:.0f}")

# The mean is the best single guess under squared error, the
# median under absolute error.
print("\nOne guess for every contract, and what it costs")
print(f"{'guess':>14}{'squared error':>15}{'absolute error':>16}")
for name, c in (("mean", mean), ("median", median), ("30 days", 30)):
    print(f"{name:>14}{((gap - c) ** 2).mean():>15.1f}"
          f"{np.abs(gap - c).mean():>16.2f}")

spend = train.spend_365
z = (spend - spend.mean()) / spend.std()
print("\nSpend in the year before the mark, dollars")
print(f"  mean {spend.mean():,.0f}, median {spend.median():,.0f},"
      f" standard deviation {spend.std():,.0f}")
print(f"  largest row {spend.max():,.0f}, {z.max():.1f}"
      " standard deviations above the mean")
cubed = (z ** 3).mean()
print(f"  skewness by the formula {cubed:.1f}, by pandas"
      f" {spend.skew():.1f}")
logged = np.log10(spend[spend > 0])
print(f"  on a log10 scale: mean {logged.mean():.2f}"
      f" (= {10 ** logged.mean():,.0f} dollars), skewness"
      f" {logged.skew():.2f}")
