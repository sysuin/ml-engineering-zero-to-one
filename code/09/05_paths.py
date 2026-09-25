# Ridge and lasso paths: every weight as the leash tightens.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, known_by
from foresight.models.logistic import TRAIN
from foresight.models.regularised import RegularisedRisk

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
strengths = np.geomspace(1e-5, 0.1, 49)
paths = {pen: pd.DataFrame([RegularisedRisk(pen, s).fit(train)
                            .weights() for s in strengths],
                           index=strengths)
         for pen in ("l2", "l1")}
free = RegularisedRisk("l2", 0.0).fit(train).weights()
lasso = paths["l1"]
zero = (lasso == 0).to_numpy()[::-1]    # strongest leash first
stays = np.logical_and.accumulate(zero)[::-1]   # zero from here on
gone = {c: strengths[stays[:, j]].min() for j, c in enumerate(lasso)}
print(f"{'column':<24}{'unpenalised':>12}{'lasso zero from':>17}")
for c in sorted(gone, key=lambda c: -gone[c]):
    print(f"{c:<24}{free[c]:>+12.3f}{gone[c]:>17.5f}")
for s in (0.001, 0.002, 0.005):
    i = np.argmin(abs(strengths - s))
    ridge, kept = paths["l2"].iloc[i], (lasso.iloc[i] != 0).sum()
    print(f"strength {strengths[i]:.4f}: lasso keeps {kept} of 16;"
          f" ridge's largest |w| {ridge.abs().max():.3f}")

with open("code/09/05_paths.json", "w") as f:
    json.dump({"strengths": strengths.tolist(),
               "columns": list(free.index),
               **{p: d.to_numpy().T.tolist()
                  for p, d in paths.items()}}, f)
