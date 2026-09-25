# Which cells are missing, why, and what the gaps say about leaving.
import json

import pandas as pd

from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
missing = train.isna().sum()
print("Missing values in the training split")
for col, n in missing[missing > 0].items():
    print(f"  {col:18}{n:>6,}  {n / len(train):6.1%}")

gap = train.discount_pct.isna()
print("\nDiscount missing, against legacy terms (rows)")
print(pd.crosstab(train.legacy_terms, gap,
                  colnames=["discount missing"]).to_string())

since = train.moment - pd.to_timedelta(train.tenure_days, unit="D")
for flag, g in since.groupby(train.legacy_terms):
    print(f"  legacy_terms {flag}: customers since {g.min():%Y-%m} to"
          f" {g.max():%Y-%m}")

print("\nNot renewed")
for name, rows in [("discount missing", train[gap]),
                   ("discount recorded", train[~gap])]:
    print(f"  {name:18}{rows.not_renewed.sum():>4} of"
          f" {len(rows):>5,}  {rows.not_renewed.mean():.1%}")

# Three things people do with a gap, and what each does to it.
leavers = train.not_renewed.sum()
kept = train[~gap]
lost = leavers - kept.not_renewed.sum()
print(f"\nDrop the rows: {len(kept):,} left; {lost} of {leavers}"
      f" leavers lost ({lost / leavers:.0%});"
      f" rate {kept.not_renewed.mean():.1%}")
zero = train.discount_pct.fillna(0)
print(f"Fill with 0: 'no discount' now {(zero == 0).sum():,} rows,"
      f" {train.not_renewed[zero == 0].mean():.1%} not renewed")
real_zero = train[train.discount_pct == 0]
print(f"  the {len(real_zero):,} real zeros alone:"
      f" {real_zero.not_renewed.mean():.1%}")
mean = train.discount_pct.mean()
print(f"Fill with the mean: {gap.sum():,} rows get {mean:.1f}%,"
      f" a discount no contract has")

# For the figure: the missing cells, row by row, in two orders.
cols = list(train.columns)
by_moment = train.isna()[cols]
by_legacy = train.sort_values(["legacy_terms", "moment"]).isna()[cols]
with open("code/05/04_missingness.json", "w") as f:
    json.dump({"columns": cols,
               "by_moment": {c: by_moment[c].to_numpy().nonzero()[0]
                             .tolist() for c in cols},
               "by_legacy": {c: by_legacy[c].to_numpy().nonzero()[0]
                             .tolist() for c in cols},
               "legacy_from": int((train.legacy_terms == 0).sum()),
               "rows": len(train)}, f)
