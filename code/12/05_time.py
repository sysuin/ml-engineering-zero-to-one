# Time as features: days to renewal, tenure, the month of the mark, and
# how many days of history lie behind each row's windows.
import numpy as np

from foresight.features.build import load
from foresight.features.sources import load_sources
from foresight.models.featured import TRAINING

table = load()
train = table[table.end_date.between(*TRAINING)]
y = train.not_renewed


def rates(groups, title):
    g = y.groupby(groups, observed=True).agg(["size", "mean"])
    print(f"\n{title:<26}{'contracts':>10}{'left':>8}{'margin':>9}")
    for k, r in g.iterrows():
        m = 1.96 * np.sqrt(r["mean"] * (1 - r["mean"]) / r["size"])
        print(f"  {str(k):<24}{r['size']:>10,.0f}{r['mean']:>8.1%}"
              f"{m:>8.1%}")


days_left = (train.end_date - train.moment).dt.days.value_counts()
print(f"Days from mark to renewal: {days_left.index[0]} for"
      f" {days_left.iloc[0]:,} of {len(train):,} contracts")

years = train.tenure_days // 365 + 1
rates(years.clip(upper=6).map(lambda n: f"renewal {n}" if n < 6
                              else "renewal 6 or later"),
      "Training, by tenure")
rates(train.moment.dt.quarter.map(lambda q: f"mark in Q{q}"),
      "Training, by season")

short = train.days_of_history < 365
keyed = load_sources().accounts.set_index("account_id").is_key_account
key = train.account_id.map(keyed) == 1
print(f"\nWindows not fully on record: {short.sum()} of {len(train):,}"
      f" training rows")
for name, rows in (("long tail", short & ~key),
                   ("key accounts", short & key)):
    d = train.days_of_history[rows]
    print(f"  {name:<13}{rows.sum():>4} rows, median {d.median():>4.0f}"
          f" days on record, {(d == 0).sum():>3} with none")
print(f"No order at all before the mark: "
      f"{int(train.no_order_record.sum())} rows")
