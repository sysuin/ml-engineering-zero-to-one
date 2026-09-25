# Exercise 5: leavers counted by row, and by the spend they took away.
import pandas as pd

from foresight.data.build_table import TABLE

train = pd.read_parquet(TABLE).query("end_date <= '2024-06-30'")
train = train.assign(lost=train.spend_365 * train.not_renewed)
by = train.groupby("region").agg(rows=("not_renewed", "size"),
                                 left=("not_renewed", "sum"),
                                 spend=("spend_365", "sum"),
                                 lost=("lost", "sum"))
by["by_row"] = by.left / by.rows
by["by_money"] = by.lost / by.spend
print(f"{'':11}{'rows':>6}{'left':>6}{'by row':>9}{'by money':>10}")
for region, r in by.iterrows():
    print(f"{region:11}{r.rows:>6,.0f}{r.left:>6,.0f}{r.by_row:>9.1%}"
          f"{r.by_money:>10.1%}")
mid = train[train.region == "Midwest"]
top = mid.lost.max()
share = top / mid.lost.sum()
print(f"\nMidwest: its largest single loss is {share:.0%} of the"
      " region's lost spend")
