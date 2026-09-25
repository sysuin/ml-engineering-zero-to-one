# One column, thirty contracts, and polynomials of rising degree.
import json

import numpy as np
from numpy.polynomial import Polynomial

from foresight.config import rng
from foresight.data.spend import (TRAIN, VALIDATION, spend_table,
                                  year_on_record)

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
train = train[year_on_record(train)                  # as in Chapter 6
              & (train.end_date < valid.moment.min())]   # no gap
x, y = np.log1p(train.spend_90d), np.log1p(train.spend_next_90d)
xv, yv = np.log1p(valid.spend_90d), np.log1p(valid.spend_next_90d)
x, y, xv, yv = (a.to_numpy() for a in (x, y, xv, yv))
print(f"pool of {len(x):,} training contracts; validation {len(xv):,}")


def rmse(fit, xs, ys):
    return float(np.sqrt(np.mean((fit(xs) - ys) ** 2)))


DEGREES = range(9)
g = rng()
draws = [g.choice(len(x), 30, replace=False) for _ in range(200)]
errors = np.zeros((len(draws), len(DEGREES), 2))
fits = {d: [] for d in DEGREES}
for r, ix in enumerate(draws):
    for d in DEGREES:
        fit = Polynomial.fit(x[ix], y[ix], d)
        errors[r, d] = rmse(fit, x[ix], y[ix]), rmse(fit, xv, yv)
        fits[d].append(fit)

print("\nRMSE of log(1 + spend) after the mark")
print(f"{'degree':>6}{'first 30: train':>17}{'valid':>11}"
      f"{'200 draws: train':>18}{'valid':>11}")
for d in DEGREES:
    (a, b), (c, e) = errors[0, d], np.median(errors[:, d], axis=0)
    print(f"{d:>6}{a:>17.2f}{b:>11.2f}{c:>18.2f}{e:>11.2f}")
full = {d: rmse(Polynomial.fit(x, y, d), xv, yv) for d in (0, 2, 8)}
print(f"\nfitted on all {len(x):,} instead, validation RMSE")
print("  " + "   ".join(f"degree {d} {e:.2f}"
                        for d, e in full.items()))

grid = np.linspace(0, x.max(), 120)
with open("code/09/02_three_fits.json", "w") as f:
    json.dump({"x": x[draws[0]].tolist(), "y": y[draws[0]].tolist(),
               "grid": grid.tolist(),
               "curves": {d: [fits[d][r](grid).tolist()
                              for r in range(21)] for d in (0, 2, 8)},
               "median": np.median(errors, axis=0).tolist(),
               "first": errors[0].tolist()}, f)
