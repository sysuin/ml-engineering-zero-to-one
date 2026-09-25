# Residuals of the nine-feature line: their summary, shape and extremes.
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from foresight.config import ML_WAREHOUSE
from foresight.data.spend import (TRAIN, VALIDATION, features,
                                  spend_table)

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
model = LinearRegression().fit(features(train),
                               train.spend_next_90d)
valid["pred"] = model.predict(features(valid))
valid["resid"] = valid.spend_next_90d - valid.pred  # actual - predicted

r = valid.resid
print(f"Validation residuals, {len(r):,} rows (dollars)")
print(f"  mean {r.mean():>8,.0f}   median {r.median():>8,.0f}"
      f"   above zero {(r > 0).mean():.1%}")
print("  percentiles " + "  ".join(
    f"{q:.0%} {r.quantile(q):,.0f}" for q in (0.05, 0.25, 0.75, 0.95)))
print(f"  predictions below zero: {(valid.pred < 0).sum()}")

bands = pd.cut(valid.pred, [-np.inf, 2_000, 5_000, 10_000, 50_000,
                            np.inf])
print(f"\n{'predicted':<20}{'rows':>6}{'mean resid':>12}"
      f"{'mean |resid|':>14}")
for band, g in valid.groupby(bands, observed=True):
    lo, hi = band.left, band.right
    name = (f"under {hi:,.0f}" if lo == -np.inf else
            f"{lo:,.0f} and up" if hi == np.inf else
            f"{lo:,.0f} to {hi:,.0f}")
    print(f"{name:<20}{len(g):>6,}{g.resid.mean():>12,.0f}"
          f"{g.resid.abs().mean():>14,.0f}")

# The training rows the line fits worst, and who they are.
train["resid"] = train.spend_next_90d - model.predict(features(train))
with sqlite3.connect(ML_WAREHOUSE) as con:
    names = pd.read_sql_query("SELECT account_id, name FROM accounts",
                              con)
worst = (train.merge(names, on="account_id")
         .assign(size=lambda d: d.resid.abs())
         .nlargest(6, "size"))
print(f"\n{'worst training rows':<22}{'mark':>11}{'key':>4}"
      f"{'spend_365':>11}{'actual':>10}{'resid':>10}")
for _, w in worst.iterrows():
    print(f"{w['name']:<22} {w.moment:%Y-%m-%d}{w.is_key_account:>4}"
          f"{w.spend_365:>11,.0f}{w.spend_next_90d:>10,.0f}"
          f"{w.resid:>10,.0f}")

with open("code/06/08_residuals.json", "w") as f:
    json.dump({"pred": valid.pred.round(1).tolist(),
               "resid": valid.resid.round(1).tolist()}, f)
