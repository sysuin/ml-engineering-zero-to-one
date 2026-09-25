# The support queue in numbers: volume, labels, languages and splits.
import pandas as pd

from foresight.triage.evaluate import (CAPACITY, keyword_rule,
                                       urgent_at_capacity)
from foresight.triage.features import split, tickets

t = tickets()
per_day = t.groupby("day").size()
print(f"{len(t):,} tickets, {t.day.min():%Y-%m-%d} to "
      f"{t.day.max():%Y-%m-%d}")
last = per_day[per_day.index >= "2025-01-01"]
print(f"2025: {last.mean():.1f} a day (median {last.median():.0f}, "
      f"{last.min()} to {last.max()})")

shares = pd.DataFrame({
    "priority": t.priority.value_counts(normalize=True),
    "category": t.category.value_counts(normalize=True)})
print("\nlabel shares, as the desks filed them")
for name in ["priority", "category"]:
    s = shares[name].dropna().sort_values(ascending=False)
    items = [f"{k} {v:.1%}" for k, v in s.items()]
    for i in range(0, len(items), 4):
        head = name if i == 0 else ""
        print(f"  {head:9}" + "  ".join(items[i:i + 4]))
langs = t.language.value_counts()
print("  language " + "  ".join(f"{k} {v:,}" for k, v in langs.items()))

train, valid, test = split(t)
print("\nsplit by the day a ticket was opened")
for name, part in [("train", train), ("validation", valid),
                   ("test", test)]:
    print(f"  {name:11}{part.day.min():%Y-%m-%d} to "
          f"{part.day.max():%Y-%m-%d}  {len(part):6,} tickets")

# Today's practice, as scores: arrival order, and the urgent words.
arrival = -valid.opened_at.astype("int64")
rule = valid.body.map(keyword_rule)
print(f"\nvalidation: {(valid.priority == 'Urgent').sum()} Urgent; "
      f"reading the first {CAPACITY} a day finds")
print(f"  in order of arrival   "
      f"{urgent_at_capacity(valid, arrival):.1%}")
print(f"  urgent words first    {urgent_at_capacity(valid, rule):.1%}")
busy = valid.groupby("day").size().mean()
print(f"  {CAPACITY} a day is {CAPACITY / busy:.1%} of the {busy:.1f} "
      f"tickets an average day brought")
