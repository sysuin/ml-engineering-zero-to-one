# Reliability tables for v0.4's lasso and the booster on validation:
# contracts sorted by score, cut into ten bins of equal size, and the
# average chance given against the share that left.
import json

import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.decide import calibration_slope, reliability
from foresight.evaluate import SPLITS, backtest, interval, resample
from foresight.models.boosting import RenewalBooster
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
runs = {"lasso": maker("l1", 0.002), "booster": RenewalBooster}
tables, saved = {}, {}
for name, make in runs.items():
    s = backtest(table, *SPLITS["validation"], make)
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    tables[name] = reliability(p, y)
    # Each bin's share that left, on 2,000 resamples of contracts
    # within cohorts, Chapter 8's bootstrap.
    order = np.argsort(p, kind="stable")
    for i, part in enumerate(np.array_split(order, 10)):
        s.loc[part, "bin"] = i
    g = rng()
    shares = pd.DataFrame([resample(s, g).groupby("bin").not_renewed
                           .mean() for _ in range(2000)])
    slope, intercept = calibration_slope(p, y)
    saved[name] = {**tables[name].to_dict("list"),
                   "range": [interval(shares[i]) for i in range(10)],
                   "slope": slope, "intercept": intercept}

lasso, boost = tables["lasso"], tables["booster"]
print(f"{'':6}{'lasso':>25}{'booster':>24}")
print(f"{'bin':<6}{'contracts':>9}{'said':>8}{'left':>8}"
      f"{'contracts':>12}{'said':>8}{'left':>8}")
for (_, a), (_, b) in zip(lasso.iterrows(), boost.iterrows()):
    print(f"{int(a.bin):<6}{int(a.contracts):>9}{a.predicted:>8.1%}"
          f"{a.left:>8.1%}{int(b.contracts):>12}{b.predicted:>8.1%}"
          f"{b.left:>8.1%}")
print(f"\n{'':<9}{'slope':>6}{'intercept':>11}{'reliability':>13}"
      f"{'resolution':>12}{'uncertainty':>13}")
for name in runs:
    t = tables[name]
    n = t.contracts.sum()
    rate = (t.left * t.contracts).sum() / n
    rel = ((t.predicted - t.left) ** 2 * t.contracts).sum() / n
    res = ((t.left - rate) ** 2 * t.contracts).sum() / n
    print(f"{name:<9}{saved[name]['slope']:>6.2f}"
          f"{saved[name]['intercept']:>+11.2f}{rel:>13.4f}"
          f"{res:>12.4f}{rate * (1 - rate):>13.4f}")

with open("code/14/07_reliability.json", "w") as f:
    json.dump(saved, f)
