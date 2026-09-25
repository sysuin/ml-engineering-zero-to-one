# "They slow down before they leave": the hunch as one column, checked
# on the training period, then read once on the validation cohorts.
# timeout: 300
import pandas as pd

from foresight.evaluate import SPLITS, backtest
from foresight.features.build import load
from foresight.models.featured import (TRAINING, FeaturedLasso, booster,
                                       SHOWN, gain, lasso, shown,
                                       tuning_score, watched_loss)

table = load()                          # Chapter 4's rows + the library
train = table[table.end_date.between(*TRAINING)]

# Is it true? The share that left, by how orders moved quarter on
# quarter: order_trend = log((last 90 days + 1) / (the 90 before + 1)).
bands = pd.cut(train.order_trend, [-9, -1, -0.3, 0.3, 1, 9],
               labels=["fell by over 60%", "fell", "steady", "rose",
                       "rose by over 170%"])
rates = train.groupby(bands, observed=True).not_renewed.agg(
    ["size", "mean"])
print(f"{'Training contracts':<22}{'contracts':>10}{'left':>8}")
for band, r in rates.iterrows():
    print(f"  orders {band:<18}{r['size']:>8,.0f}{r['mean']:>8.1%}")

# Did v0.4 already know? The lasso's weights on the two order counts.
w = FeaturedLasso().fit(train).weights()
print(f"\nv0.4's weights: orders_90d {w.orders_90d:+.3f},"
      f" orders_prev_90d {w.orders_prev_90d:+.3f}")

# What it is worth on the training period, beside the settings
# Chapters 9 and 11 searched there.
print(f"\n{'Training period':<34}{'log loss':>9}{'AUC':>7}"
      f"{'leavers':>9}")
for name, make in [("lasso, no penalty", lasso((), 0.0)),
                   ("lasso, Chapter 9's penalty", lasso()),
                   ("  + order_trend", lasso(["order_trend"]))]:
    s = tuning_score(table, make)
    print(f"  {name:<32}{s['log loss']:>9.5f}{s['auc']:>7.3f}"
          f"{s['hits']:>9}")
print(f"{'':<34}{'watched':>9}{'rounds':>9}")
for name, extra, shape in [
        ("booster, LightGBM's 31 leaves", (), (31, 20)),
        ("booster, Chapter 11's stumps", (), (2, 50)),
        ("  + order_trend", ["order_trend"], (2, 50))]:
    loss, rounds = watched_loss(table, extra, *shape)
    print(f"  {name:<32}{loss:>9.5f}{rounds:>9,}")

# Validation, read once: each model with and without the column.
print("\nValidation, with minus without, 95% intervals")
print(f"{'':10}{SHOWN}")
for name, make in (("lasso", lasso), ("booster", booster)):
    old = backtest(table, *SPLITS["validation"], make())
    new = backtest(table, *SPLITS["validation"], make(["order_trend"]))
    print(f"  {name:<11}{shown(gain(new, old))}")
