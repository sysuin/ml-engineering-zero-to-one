# Fraud labels arrive late: what a training table sees, and when.
import json
from pathlib import Path

import numpy as np

from foresight.config import rng

# Assumed: how long after the transaction a fraud is reported. Most
# cardholders notice on a statement; some take weeks; the last few
# chargebacks come in near the scheme's limit of about 120 days.
g = rng()
n = 200_000
kind = g.choice(3, size=n, p=[0.55, 0.35, 0.10])
delay = np.where(kind == 0, g.lognormal(np.log(12), 0.6, n),
                 np.where(kind == 1, g.lognormal(np.log(35), 0.5, n),
                          g.uniform(60, 120, n)))
delay = np.minimum(delay, 120)


def known(age_days) -> float:
    """Share of eventual frauds already reported at this age."""
    return float((delay <= age_days).mean())


print("Frauds already reported, by the transaction's age")
ages = [7, 14, 30, 60, 90, 120]
print("  age, days " + "".join(f"{a:>7}" for a in ages))
print("  reported  " + "".join(f"{known(a):>7.0%}" for a in ages))

# The naive table: the last 90 days, labelled with what is known today.
days = np.arange(90)
seen = np.array([known(a) for a in days])
print(f"\nLast 90 days, labels as of today: {seen.mean():.0%} of the"
      " frauds\nare labelled fraud; the rest are labelled legitimate.")
print(f"Newest fortnight: {seen[:14].mean():.0%} of its frauds known.")

# The fix: train on a window that ends W days ago, so its labels have
# had W days to mature. The price is a model W days older.
print(f"\n{'wait W':>8}{'labels complete':>17}{'newest row':>12}")
for wait in (0, 30, 60, 90):
    ages_w = days + wait
    done = np.mean([known(a) for a in ages_w])
    print(f"{wait:>6} d{done:>17.1%}{wait:>9} d old")

Path("code/26/02_label_delay.json").write_text(json.dumps(
    {"age": list(range(0, 121)),
     "known": [known(a) for a in range(0, 121)]}, indent=1))
