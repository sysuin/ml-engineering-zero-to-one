# Exercise 4: a second condition for the gate. The challenger's log
# loss on the six cohorts may be no worse than the champion's; shown
# as a point and as a paired interval, for listing 23/08's branches.
import numpy as np
import pandas as pd

from foresight import gate
from foresight.config import SEED, rng
from foresight.data.build_table import TABLE
from foresight.evaluate import REPS, interval
from foresight.models.logistic import log_loss

table = pd.read_parquet(TABLE)
main, cfg = gate.candidate()
as_of, n = cfg["data"]["as_of"], cfg["evaluation"]["cohorts"]
champion = gate.scores(main, table, as_of, n)
no_region, _ = gate.candidate()
no_region.set_params(estimator__prepare__region="drop")
branches = {"a tidy-up of the code": gate.candidate()[0],
            "strength 0.002 -> 0.02":
                gate.candidate(["model.strength=0.02"])[0],
            "region dropped": no_region}

y = champion.not_renewed.to_numpy()
groups = list(champion.groupby("moment").indices.values())
print(f"{'branch':<24}{'log loss':>9}{'change':>9}{'(95%)':>20}")
print(f"{'v0.6':<24}{log_loss(y, champion.model):>9.4f}")
for name, model in branches.items():
    s = gate.scores(model, table, as_of, n)
    a, b = s.model.to_numpy(), champion.model.to_numpy()
    g, d = rng(SEED), []
    for _ in range(REPS):
        i = np.concatenate([r[g.integers(0, len(r), len(r))]
                            for r in groups])
        d.append(log_loss(y[i], a[i]) - log_loss(y[i], b[i]))
    lo, hi = interval(d)
    diff = log_loss(y, a) - log_loss(y, b)
    print(f"{name:<24}{log_loss(y, a):>9.4f}{diff:>+9.4f}"
          f"{f'{lo:+.4f} to {hi:+.4f}':>20}")
