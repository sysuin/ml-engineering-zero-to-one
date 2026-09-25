# Predicting log(1 + spend) instead of spend: residuals, dollar scores.
import json

import numpy as np
from sklearn.linear_model import LinearRegression

from foresight.data.spend import (TRAIN, VALIDATION, baselines,
                                  features, scores, spend_table,
                                  year_on_record)

everything = spend_table(*TRAIN)
train = everything[year_on_record(everything)]       # as in 09
valid = spend_table(*VALIDATION)
y, v = train.spend_next_90d.to_numpy(), valid.spend_next_90d.to_numpy()


def logged(rows):
    """The five heavy-tailed columns on a log scale; the rest as are."""
    X = features(rows)
    X[:, :5] = np.log1p(X[:, :5])
    return X


dollars = LinearRegression().fit(features(train), y)
logs = LinearRegression().fit(logged(train), np.log1p(y))
back = np.expm1(logs.predict(logged(valid)))       # into dollars again

res = np.log1p(v) - logs.predict(logged(valid))
print("Validation residuals on the log scale")
for name, part in (("spent something", res[v > 0]),
                   ("spent nothing", res[v == 0])):
    print(f"  {name:<16}{len(part):>6,} rows   mean {part.mean():>6.2f}"
          f"   std {part.std():>5.2f}")

print(f"\nValidation, {len(valid):,} rows")
scores(v, baselines(everything, valid) | {
    "dollars, full-year rows": dollars.predict(features(valid)),
    "log(1 + spend), back to $": back})
in_dollars = dollars.predict(features(valid))
print(f"\nsum of predictions / sum spent:"
      f"  dollars {in_dollars.sum() / v.sum():.1%}"
      f"   log {back.sum() / v.sum():.1%}")

with open("code/06/10_log_target.json", "w") as f:
    json.dump({"pred": logs.predict(logged(valid)).round(4).tolist(),
               "resid": res.round(4).tolist()}, f)
