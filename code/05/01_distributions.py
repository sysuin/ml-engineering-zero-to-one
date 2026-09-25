# Two features of the training split, summarised three ways.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE

table = pd.read_parquet(TABLE)
train = table[table.end_date <= "2024-06-30"]      # training only
print(f"{len(table):,} rows in the table; {len(train):,} in the"
      " training split")


def three_views(values, bin_width, points):
    """What a histogram, a box plot and an ECDF each report."""
    s = values.dropna().astype(float)
    edges = np.arange(0, s.max() + bin_width, bin_width)
    counts, _ = np.histogram(s, bins=edges)
    q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
    fence = q3 + 1.5 * (q3 - q1)          # the box plot's whisker rule
    print(f"  {len(s):,} values, median {median:,.0f},"
          f" mean {s.mean():,.0f}, max {s.max():,.0f}")
    print(f"  histogram: {len(counts)} bins of {bin_width:,};"
          f" the first holds {counts[0] / len(s):.1%}")
    print(f"  box: middle half {q1:,.0f} to {q3:,.0f}, whisker"
          f" {fence:,.0f}; {(s > fence).sum():,} beyond it")
    for p in points:
        print(f"  ECDF: {(s <= p).mean():6.1%} of rows at or below"
              f" {p:,}")
    return s, counts, (q1, median, q3, fence)


print("\nDays since the last order")
days, counts, box = three_views(train.days_since_order, 10,
                                [30, 90, 180])
print("\nSpend in the year before the mark ($)")
three_views(train.spend_365, 50_000, [10_000, 50_000, 200_000])

# What the box plot calls outliers, and who they are.
beyond = train[train.days_since_order > box[3]]
print(f"\nBeyond the recency whisker: {len(beyond):,} rows,"
      f" {beyond.not_renewed.mean():.1%} not renewed")
print(f"Inside it: {len(train) - len(beyond):,} rows,"
      f" {train.drop(beyond.index).not_renewed.mean():.1%} not renewed")

medians = train.groupby("not_renewed").days_since_order.median()
print(f"Median days since the last order: renewed {medians[0]:.0f},"
      f" not renewed {medians[1]:.0f}")

# For the figure: recency as a histogram, a box and two ECDFs.
grid = np.linspace(0, 1, 101)
ecdf = {name: [round(float(v), 1) for v in
               g.days_since_order.dropna().astype(float).quantile(grid)]
        for name, g in train.groupby("not_renewed")}
with open("code/05/01_distributions.json", "w") as f:
    json.dump({"bin_width": 10, "counts": counts.tolist(),
               "box": [round(v, 1) for v in box],
               "max": float(days.max()),
               "outliers": sorted(set(days[days > box[3]].tolist())),
               "ecdf_renewed": ecdf[0], "ecdf_left": ecdf[1]}, f)
