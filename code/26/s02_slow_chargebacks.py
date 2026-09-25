# Exercise 4: a fifth of fraud reported only by late chargebacks.
import numpy as np

from foresight.config import rng

g = rng()
n = 200_000


def delays(late_share):
    """The listing's mixture, with a larger share at 60-120 days."""
    rest = 1 - late_share
    kind = g.choice(3, size=n, p=[0.55 / 0.9 * rest,
                                  0.35 / 0.9 * rest, late_share])
    d = np.where(kind == 0, g.lognormal(np.log(12), 0.6, n),
                 np.where(kind == 1, g.lognormal(np.log(35), 0.5, n),
                          g.uniform(60, 120, n)))
    return np.minimum(d, 120)


days = np.arange(90)
print(f"{'late share':>10}{'wait for 95%':>14}{'known at 0 d':>14}")
for share in (0.10, 0.20):
    d = delays(share)
    known = np.array([(d <= a).mean() for a in range(0, 121)])
    complete = [known[np.minimum(days + w, 120)].mean()
                for w in range(0, 121)]
    wait = next(w for w, c in enumerate(complete) if c >= 0.95)
    print(f"{share:>10.0%}{wait:>12} d{complete[0]:>14.1%}")
