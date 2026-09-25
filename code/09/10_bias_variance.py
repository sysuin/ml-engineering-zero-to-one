# Bias and variance at one point, by simulating many training sets.
import numpy as np
from numpy.polynomial import Polynomial

from foresight.config import rng
from foresight.data.spend import (TRAIN, VALIDATION, spend_table,
                                  year_on_record)

train = spend_table(*TRAIN)
first_mark = spend_table(*VALIDATION).moment.min()
train = train[year_on_record(train) & (train.end_date < first_mark)]
x = np.log1p(train.spend_90d.to_numpy())
y = np.log1p(train.spend_next_90d.to_numpy())

# A world we control: the pool's own degree-2 curve is the truth, and
# the noise is drawn from the pool's own misses around it.
truth = Polynomial.fit(x, y, 2)
misses = y - truth(x)
x0 = np.log1p(1_000)                    # $1,000 in the last 90 days
print(f"x0 = log(1 + 1,000) = {x0:.2f}: true curve {truth(x0):.3f},"
      f" noise variance {misses.var():.3f}")

g = rng()
print(f"\n{'degree':>6}{'bias^2':>9}{'variance':>10}{'noise':>8}"
      f"{'sum':>8}{'measured':>11}")
for d in (0, 1, 2, 4, 8):
    guesses = []
    for _ in range(4_000):
        ix = g.integers(0, len(x), 30)           # 30 contracts' spend
        ys = truth(x[ix]) + g.choice(misses, 30)  # their made-up next
        guesses.append(Polynomial.fit(x[ix], ys, d)(x0))
    guesses = np.array(guesses)
    # Squared error against every possible outcome at x0, averaged.
    y0 = truth(x0) + misses
    error = np.mean((guesses[:, None] - y0[None, :]) ** 2)
    bias2 = (np.mean(guesses) - truth(x0)) ** 2
    var = np.var(guesses)
    total = bias2 + var + misses.var()
    print(f"{d:>6}{bias2:>9.3f}{var:>10.3f}{misses.var():>8.3f}"
          f"{total:>8.3f}{error:>11.3f}")
