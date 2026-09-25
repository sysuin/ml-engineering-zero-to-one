# Rows whose year before the mark is only partly on record, and a refit.
from sklearn.linear_model import LinearRegression

from foresight.data.spend import (TRAIN, VALIDATION, baselines,
                                  features, scores, spend_table,
                                  year_on_record)

train = spend_table(*TRAIN)
valid = spend_table(*VALIDATION)
full = year_on_record(train)
print(f"{'training rows':<16}{'rows':>7}{'partial year':>14}"
      f"{'last partial mark':>20}")
for key, g in train.groupby("is_key_account"):
    part = g[~full[g.index]]
    print(f"{'key accounts' if key else 'long tail':<16}{len(g):>7,}"
          f"{len(part):>14,}{'':>10}{part.moment.max():%Y-%m-%d}")
print("validation rows with a full year:"
      f" {year_on_record(valid).mean():.0%}")

X, y = features(train), train.spend_next_90d
every = LinearRegression().fit(X, y)
kept = LinearRegression().fit(X[full], y[full])
v = valid.spend_next_90d
print(f"\nValidation, {len(valid):,} rows")
scores(v, baselines(train, valid) | {
    "nine features, all rows": every.predict(features(valid)),
    f"nine features, {full.sum():,} full-year": kept.predict(
        features(valid))})
below = (kept.predict(features(valid)) < 0).sum()
print(f"predictions below zero: {below}")
