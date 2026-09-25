# Exercise 4: the simplest strength within one standard error of best.
import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import backtest
from foresight.models.logistic import log_loss
from foresight.models.regularised import STRENGTHS, TUNE, maker

table = pd.read_parquet(TABLE)
runs = {s: backtest(table, *TUNE, maker("l1", s)) for s in STRENGTHS}
y = runs[0.0].not_renewed.to_numpy()
p = {s: r.model.to_numpy() for s, r in runs.items()}
loss = {s: log_loss(y, v) for s, v in p.items()}
best = min(loss, key=loss.get)

# Standard error of each strength's gap to the best, by resampling
# contracts within cohorts, as Chapter 8 does.
cells = runs[0.0].groupby("moment").indices.values()
g, gaps = rng(), {s: [] for s in STRENGTHS}
for _ in range(1000):
    ix = np.concatenate([c[g.integers(0, len(c), len(c))]
                         for c in cells])
    for s in STRENGTHS:
        gaps[s].append(log_loss(y[ix], p[s][ix])
                       - log_loss(y[ix], p[best][ix]))
print(f"{'strength':>9}{'log loss':>10}{'gap to best':>13}{'SE':>8}")
for s in STRENGTHS:
    print(f"{s:>9g}{loss[s]:>10.4f}{loss[s] - loss[best]:>+13.4f}"
          f"{np.std(gaps[s]):>8.4f}")
within = [s for s in STRENGTHS
          if loss[s] - loss[best] <= np.std(gaps[s]) and s >= best]
print(f"best {best:g}; strongest within one SE: {max(within):g}")
