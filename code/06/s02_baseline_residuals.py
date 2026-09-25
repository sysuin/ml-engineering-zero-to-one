# Exercise 2: the residual table of 6.12 for each of the baselines.
import numpy as np
import pandas as pd

from foresight.data.spend import (TRAIN, VALIDATION, baselines,
                                  spend_table)

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
y = valid.spend_next_90d.to_numpy()
cuts = [-np.inf, 2_000, 5_000, 10_000, 50_000, np.inf]
names = ["under 2,000", "2,000 to 5,000", "5,000 to 10,000",
         "10,000 to 50,000", "50,000 and up"]
for name, pred in baselines(train, valid).items():
    band = pd.cut(pred, cuts, labels=names)
    frame = pd.DataFrame({"band": band, "resid": y - pred})
    print(f"{name}  (actual minus predicted)")
    print(f"  {'predicted':<20}{'rows':>6}{'mean resid':>12}"
          f"{'mean |resid|':>14}")
    for b, g in frame.groupby("band", observed=True):
        print(f"  {b:<20}{len(g):>6,}{g.resid.mean():>12,.0f}"
              f"{g.resid.abs().mean():>14,.0f}")
    print()
