# Exercise 1: the training table with the moment at 120 days, not 90.
from foresight.data import build_table as bt

FIRST, LAST = "2023-01-01", "2024-06-30"
tables = {}
for days in (90, 120):
    bt.MOMENT_DAYS = days           # the builder reads it at call time
    tables[days] = bt.build(FIRST, LAST).set_index("contract_id")
early, late = tables[120], tables[90]

print(f"Rows: {len(late):,} at 90 days, {len(early):,} at 120;"
      f" same contracts: {set(early.index) == set(late.index)}")
print("Rows whose value changed, by column")
for col in late.columns.drop(["moment"]):
    a, b = late[col], early.loc[late.index, col]
    changed = ~((a == b) | (a.isna() & b.isna()))
    print(f"  {col:18}{changed.sum():>6,}")

for days, t in ((90, late), (120, early)):
    top = (t.reset_index()
            .sort_values(["moment", "days_since_order", "contract_id"],
                         ascending=[True, False, True],
                         na_position="last")
            .groupby("moment").head(40))
    hits = top.not_renewed
    print(f"Days-since rule at {days:>3} days: {hits.sum()} leavers,"
          f" precision {hits.mean():.1%}")

# Part (c): a builder whose moment and check disagree.
bt.MOMENT_DAYS = 90
shifted = tables[120].reset_index()
print("check() on the 120-day table with MOMENT_DAYS = 90:")
for problem in bt.check(shifted, FIRST, LAST):
    print(f"  {problem}")
