# Build the training table, write it, and prove a rebuild matches.
import json

import pandas as pd

from foresight.data.build_table import TABLE, build, write

FIRST, LAST = "2023-01-01", "2025-12-31"
manifest = write(build(FIRST, LAST), FIRST, LAST)
print(f"Wrote {TABLE.name}: {manifest['rows']:,} rows,"
      f" {len(manifest['columns'])} columns")
print(f"  contracts ending {FIRST} to {LAST}")
table = pd.read_parquet(TABLE)
years = table.end_date.dt.year.value_counts().sort_index()
print("  rows by the year the contract ends")
for year, n in years.items():
    print(f"    {year:<20}{n:>6,}")
print("  missing values")
for col, n in manifest["missing"].items():
    print(f"    {col:20}{n:>6,}")
print(f"  content hash {manifest['content_sha256'][:16]}...")

again = write(build(FIRST, LAST), FIRST, LAST)
print(f"Built again from the warehouse: same content hash"
      f" {again['content_sha256'] == manifest['content_sha256']}")

# The same builder, asked for the table as it could have been built on
# 1 January 2024: only outcomes recorded by then.
then = build("2023-01-01", "2024-06-30", known_by="2024-01-01")
ends = then.end_date.dt.strftime("%Y-%m-%d")
print(f"\nAs known on 2024-01-01: {len(then):,} rows, contracts ending")
print(f"  {ends.min()} to {ends.max()}")

# The baseline rule, on the training rows of the finished table.
train = table[table.end_date <= "2024-06-30"]
ranked = train.sort_values(
    ["moment", "days_since_order", "contract_id"],
    ascending=[True, False, True], na_position="last")
top = ranked.groupby("moment").head(40)
print(f"\nDays-since rule, top 40 a cohort, {len(train):,} rows")
print(f"  {top.not_renewed.sum()} leavers in {len(top)} calls:"
      f" precision {top.not_renewed.mean():.1%}")

# For the figures: part of one cohort, and every training cohort's size.
cohort = train[train.moment == "2024-01-01"]
with open("code/04/10_build_table.json", "w") as f:
    json.dump({"columns": list(table.columns),
               "rows": cohort[cohort.contract_id.between(7330, 7335)]
               .astype(str).values.tolist(),
               "cohorts": {str(m.date()): int(n) for m, n in
                           train.moment.value_counts().sort_index()
                           .items()}},
              f, indent=1)
