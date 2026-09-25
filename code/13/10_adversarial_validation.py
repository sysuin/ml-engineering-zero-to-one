# Adversarial validation: can a model tell the training contracts from
# the validation contracts? No label is read, only the columns.
# timeout: 300
import json

import pandas as pd

from foresight.checks.leakage import CLOCK, SHIFT, adversarial
from foresight.evaluate import SPLITS
from foresight.features.build import load
from foresight.features.registry import names
from foresight.train import COLUMNS

table = load()
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
valid = table[table.end_date.between(*SPLITS["validation"])]
print(f"{len(train):,} training contracts against {len(valid):,}"
      f" validation contracts.\nFail at {CLOCK}, review from {SHIFT}\n")

out = {}
for name, cols in (("v0.5's columns", COLUMNS),
                   ("with the library", COLUMNS + names()),
                   ("with contract_id", COLUMNS + ["contract_id"])):
    together, each = adversarial(train, valid, cols)
    out[name] = {"together": together,
                 "each": each["shift auc"].round(4).to_dict()}
    flagged = each[each["shift"] != "pass"].sort_values(
        "shift auc", ascending=False)
    print(f"{name}: {len(cols)} columns, together AUC {together:.3f}")
    for c, r in flagged.iterrows():
        print(f"  {c:<20}{r['shift auc']:>7.3f}  {r['shift']}")

# Why tenure_days: the days past a whole number of years.
both = pd.concat([train.assign(split="training"),
                  valid.assign(split="validation")])
extra = (both.tenure_days % 365).where(lambda d: d.between(274, 278))
print("\nDays past a whole year  " + "".join(
    f"{d:>6}" for d in range(274, 279)))
for split, g in extra.groupby(both.split):
    share = g.value_counts(normalize=True).reindex(range(274, 279),
                                                   fill_value=0)
    print(f"  {split:<22}" + "".join(f"{v:>6.0%}" for v in share))
with open("code/13/10_adversarial_validation.json", "w") as f:
    json.dump(out, f)
