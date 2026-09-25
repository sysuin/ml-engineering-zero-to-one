# Tickets by topic: complaints by the desk's category, chasing by the
# words in the body, and products named in the body in any case.
import pandas as pd

from foresight.features.build import load
from foresight.features.definitions import COMPLAINTS, SKU
from foresight.features.sources import Context, load_sources
from foresight.models.featured import TRAINING

table = load()
train = table[table.end_date.between(*TRAINING)]
y = train.not_renewed

# Complaints alone, then within bands of account size.
some = train.complaints_90d > 0
print(f"Training: left with a complaint in 90 days {y[some].mean():.1%}"
      f" ({some.sum():,}),\n  without {y[~some].mean():.1%}")
size = pd.qcut(train.orders_365.rank(method="first"), 3,
               labels=["fewest orders", "middle third", "most orders"])
print(f"\n{'Training, by size':<20}{'complaint rate':>15}"
      f"{'left, none':>12}{'left, 1+':>10}")
for band, rows in train.groupby(size, observed=True):
    c = rows.complaints_90d > 0
    print(f"  {band:<18}{c.mean():>15.1%}"
          f"{rows.not_renewed[~c].mean():>12.1%}"
          f"{rows.not_renewed[c].mean():>10.1%}")

# The same windows, read two ways: the sku column, and the body.
ctx = Context(train, load_sources())
t = ctx.window("tickets", 90)
by_column = t.sku.notna().sum()
by_body = t.body.str.contains(SKU).sum()
print(f"\nTickets in the training rows' 90-day windows: {len(t):,}")
print(f"  naming a product, by the sku column: {by_column:,}")
print(f"  naming a product, read from the body: {by_body:,}")
print(f"  filed as {' or '.join(COMPLAINTS)}: "
      f"{t.category.isin(COMPLAINTS).sum():,}")

print(f"\n{'Training: left (rows)':<22}{'none':>14}{'one':>14}"
      f"{'two or more':>14}")
for c in ["complaints_90d", "chasing_90d", "product_tickets_90d",
          "tickets_90d"]:
    g = y.groupby(train[c].clip(upper=2)).agg(["mean", "size"])
    cells = [f"{r['mean']:.1%} ({r['size']:,.0f})"
             for _, r in g.iterrows()]
    print(f"  {c:<20}" + "".join(f"{x:>14}" for x in cells))
