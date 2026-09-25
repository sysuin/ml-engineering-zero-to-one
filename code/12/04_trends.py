# Four ways to say "slowing down": two ratios of this quarter to the
# last, a slope through a year of months, and the quarter's share.
from foresight.evaluate import auc
from foresight.features.build import load
from foresight.models.featured import TRAINING

table = load()
train = table[table.end_date.between(*TRAINING)]
y = train.not_renewed
trends = ["order_trend", "spend_trend", "spend_slope", "quarter_share"]

print(f"{'':34}{'median':>9}{'median,':>9}")
print(f"{'Training, one column alone':<28}{'AUC':>6}{'all':>9}"
      f"{'leavers':>9}")
for c in trends:
    a = auc(y, train[c])
    print(f"  {c:<26}{max(a, 1 - a):>6.3f}{train[c].median():>9.3f}"
          f"{train.loc[y == 1, c].median():>9.3f}")

# A ratio needs a denominator, and quarters can be empty.
now, before = train.orders_90d, train.orders_prev_90d
print("\nPlain ratio of order counts, last quarter over the one"
      " before:")
print(f"  {((before == 0) & (now > 0)).sum():>4} contracts divide by"
      f" zero (none before, some since)")
print(f"  {((before > 0) & (now == 0)).sum():>4} have a ratio of 0,"
      f" whose log is minus infinity")
print(f"  {((before == 0) & (now == 0)).sum():>4} have 0 over 0;"
      f" order_trend adds 1 to each count")

# How far the four agree with each other and with the two counts.
short = {"order_trend": "orders", "spend_trend": "spend",
         "spend_slope": "slope", "quarter_share": "share",
         "orders_90d": "o_90d", "orders_prev_90d": "o_prev"}
r = train[list(short)].corr("spearman").rename(index=short,
                                              columns=short)
print("\nRank correlation")
print(f"{'':10}" + "".join(f"{c:>8}" for c in r.columns))
for name, row in r.iloc[:4].iterrows():
    print(f"  {name:<8}" + "".join(f"{v:>8.2f}" for v in row))
