# Each renewal's segment on the morning of its mark, and who left.
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import (RenewalRisk, at_capacity,
                                       days_since_rule, features)
from foresight.segments import (SEGMENT_DATE, account_features,
                                fit_named, load)

orders, accounts = load()
seg = fit_named(account_features(SEGMENT_DATE, orders, accounts))

rows = pd.read_parquet(TABLE).query("end_date <= '2024-12-31'")
parts = []
for moment, cohort in rows.groupby("moment"):      # point in time
    names = seg.label(account_features(moment, orders, accounts))
    parts.append(cohort.assign(behaviour=cohort.account_id.map(names)))
rows = pd.concat(parts)
rows["behaviour"] = rows.behaviour.fillna("No order in the year")
rows["split"] = rows.end_date.le("2024-06-30").map(
    {True: "train", False: "validation"})

rate = rows.pivot_table(index="behaviour", columns="split",
                        values="not_renewed", aggfunc=["size", "mean"])
print(f"{'':21}{'train':>7}{'left':>7}{'validation':>12}{'left':>7}")
for name, r in rate.iterrows():
    print(f"{name:21}{r[('size', 'train')]:>7,.0f}"
          f"{r[('mean', 'train')]:>7.1%}"
          f"{r[('size', 'validation')]:>12,.0f}"
          f"{r[('mean', 'validation')]:>7.1%}")
total = rows.groupby("split").not_renewed.agg(["size", "mean"])
print(f"{'all renewals':21}{total.loc['train', 'size']:>7,}"
      f"{total.loc['train', 'mean']:>7.1%}"
      f"{total.loc['validation', 'size']:>12,}"
      f"{total.loc['validation', 'mean']:>7.1%}")

# How much of that the baseline rule already sees: its 40 calls a
# cohort, longest since the last order first.
val = rows[rows.split == "validation"].copy()
val["gap"] = val.days_since_order.astype("float").fillna(10_000)
calls = (val.sort_values(["moment", "gap", "contract_id"],
                         ascending=[True, False, True])
            .groupby("moment").head(40))
drift = val.behaviour == "Drifting"
print(f"\nValidation: Drifting is {drift.mean():.0%} of renewals, "
      f"{val[drift].not_renewed.sum() / val.not_renewed.sum():.0%} "
      "of leavers")
print(f"The rule's {len(calls)} calls: "
      f"{(calls.behaviour == 'Drifting').mean():.0%} to Drifting "
      "accounts")

# Chapter 7's model, with and without three segment columns.
def with_segments(r):
    X = features(r)
    for name in ["Drifting", "Newcomers", "Long-standing core"]:
        X[f"behaviour={name}"] = (r.behaviour == name).astype(float)
    return X.to_numpy()


train = rows[rows.split == "train"].sort_index()
val = rows[rows.split == "validation"].sort_index()
scale = Standardiser().fit(with_segments(train))
model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(
    scale.transform(with_segments(train)), train.not_renewed)
scores = {
    "rule": days_since_rule(val),
    "Chapter 7's model": RenewalRisk().fit(train).predict_proba(val),
    "the same + segments": model.predict_proba(
        scale.transform(with_segments(val)))[:, 1]}
print(f"\nValidation, top 40 a cohort{'leavers':>12}{'precision':>11}")
for name, score in scores.items():
    s = at_capacity(val, score)
    print(f"  {name:24}{s['leavers']:>12}{s['precision']:>11.1%}")
