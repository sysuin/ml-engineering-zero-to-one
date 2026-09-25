# Exercise 4: does a line fitted without the key accounts do better on
# the long tail?
from sklearn.linear_model import LinearRegression

from foresight.data.spend import (TRAIN, VALIDATION, baselines,
                                  features, scores, spend_table,
                                  year_on_record)

everything = spend_table(*TRAIN)
train = everything[year_on_record(everything)]
tail = train[train.is_key_account == 0]
valid = spend_table(*VALIDATION)
v_tail = valid[valid.is_key_account == 0]

both = LinearRegression().fit(features(train), train.spend_next_90d)
only = LinearRegression().fit(features(tail), tail.spend_next_90d)
print(f"Validation long tail, {len(v_tail):,} rows "
      f"(of {len(valid):,})")
scores(v_tail.spend_next_90d, baselines(everything, v_tail) | {
    f"fitted on all {len(train):,} rows":
        both.predict(features(v_tail)),
    f"fitted on {len(tail):,} long-tail": only.predict(
        features(v_tail))})
print(f"\nweight on spend_90d: {both.coef_[0]:.3f} (all),"
      f" {only.coef_[0]:.3f} (long tail)")
print(f"weight on spend_365: {both.coef_[1]:.3f} (all),"
      f" {only.coef_[1]:.3f} (long tail)")
