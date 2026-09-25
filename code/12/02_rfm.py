# Recency, frequency and monetary value: one account's three numbers,
# the analyst's RFM score, and how far the three columns get alone.
import json

import numpy as np
import pandas as pd

from foresight.evaluate import auc
from foresight.features.build import load
from foresight.features.sources import load_sources
from foresight.models.featured import (TRAINING, FeaturedLasso,
                                       tuning_score)

table = load()
train = table[table.end_date.between(*TRAINING)].copy()
train["recency"] = train.days_since_order.astype(float).fillna(365)

# One account, as the three numbers see it on the morning of its mark.
pick = train[(train.not_renewed == 1) & (train.days_of_history == 365)
             & train.orders_365.between(15, 30)
             & (train.order_trend < -0.5)].iloc[0]
orders = load_sources().orders
mine = orders[orders.account_id == pick.account_id]
age = (pick.moment - mine.day).dt.days
year = mine[(age >= 1) & (age <= 365)].assign(age=age)
print(f"Contract {pick.contract_id}, mark {pick.moment:%Y-%m-%d},"
      f" {pick.segment}, left")
print(f"  recency    {pick.recency:>9.0f} days since the last order")
print(f"  frequency  {pick.orders_365:>9.0f} orders in the year")
print(f"  monetary   {pick.spend_365:>9,.0f} dollars in the year")
print(f"  orders in windows: 30 days {pick.orders_30d:.0f}, 90 days"
      f" {pick.orders_90d}, the 90 before {pick.orders_prev_90d}")

# Each column alone, ranking the training contracts: AUC, oriented so
# that 0.5 is no better than chance whichever way the column points.
alone = {c: auc(train.not_renewed, train[c].astype(float).fillna(0))
         for c in ["recency", "orders_365", "spend_365", "orders_90d",
                   "tickets_90d", "tenure_days", "term_months"]}
print(f"\n{'One column alone, training':<30}{'AUC':>6}  direction")
for c, a in sorted(alone.items(), key=lambda kv: -abs(kv[1] - 0.5)):
    way = "higher, riskier" if a > 0.5 else "lower, riskier"
    print(f"  {c:<28}{max(a, 1 - a):>6.3f}  {way}")

# The analyst's RFM score: quintiles cut on the training rows, 1 for
# the riskiest fifth of each column to 5 for the safest, then summed.
cuts = {}
for col, risky_high in (("recency", True), ("orders_365", False),
                        ("spend_365", False)):
    q = train[col].rank(method="first", ascending=not risky_high)
    cuts[col] = pd.qcut(q, 5, labels=False) + 1
score = sum(cuts.values())
band = pd.cut(score, [2, 6, 9, 12, 15],
              labels=["3-6", "7-9", "10-12", "13-15"])
rates = train.groupby(band, observed=True).not_renewed.agg(
    ["size", "mean"])
print(f"\n{'RFM score, training':<22}{'contracts':>10}{'left':>8}")
for b, r in rates.iterrows():
    print(f"  {b:<20}{r['size']:>10,.0f}{r['mean']:>8.1%}")

# Three columns against v0.4's sixteen, on the tuning cohorts.
three = FeaturedLasso(["orders_365"], drop=(
    "orders_", "tickets", "tenure", "term", "legacy", "discount",
    "segment", "region"))
print(f"\n{'Tuning cohorts':<26}{'columns':>8}{'log loss':>10}"
      f"{'AUC':>7}{'leavers':>9}")
for name, make, n in (("recency, freq., money", lambda: three, 3),
                      ("v0.4's lasso", FeaturedLasso, 16)):
    s = tuning_score(table, make)
    print(f"  {name:<24}{n:>8}{s['log loss']:>10.5f}{s['auc']:>7.3f}"
          f"{s['hits']:>9}")

with open("code/12/02_rfm.json", "w") as f:
    json.dump({"contract": int(pick.contract_id),
               "mark": f"{pick.moment:%Y-%m-%d}",
               "age": year.age.tolist(),
               "value": np.round(year.value, 2).tolist(),
               "recency": float(pick.recency),
               "frequency": int(pick.orders_365),
               "monetary": float(pick.spend_365)}, f)
