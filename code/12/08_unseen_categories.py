# Sanitation, launched on 1 April 2024: a category the training rows
# never saw, as a one-hot column and as a share of spend.
import warnings

from sklearn.preprocessing import OneHotEncoder

from foresight.evaluate import HISTORY_FROM, SPLITS, known_by
from foresight.features.build import load
from foresight.models.featured import TRAINING, FeaturedLasso

table = load()
top = table.filter(like="share_").idxmax(axis=1)
top = top.str.removeprefix("share_").str.capitalize()
table["main_category"] = top.where(table.spend_365 > 0, "none")
train = table[table.end_date.between(*TRAINING)]
valid = table[table.end_date.between(*SPLITS["validation"])]

for name, rows in (("training", train), ("validation", valid)):
    some = (rows.share_sanitation > 0).mean()
    main = (rows.main_category == "Sanitation").sum()
    print(f"{name:<11} buying Sanitation {some:>6.1%};"
          f" as the top category: {main}")

# An encoder fitted on the training rows meets its first Sanitation.
enc = OneHotEncoder(handle_unknown="error").fit(
    train[["main_category"]])
try:
    enc.transform(valid[["main_category"]])
except ValueError as e:
    first = str(e).split(" in column")[0]
    print(f"\nhandle_unknown='error': ValueError\n  {first}")
enc = OneHotEncoder(handle_unknown="ignore").fit(
    train[["main_category"]])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")      # it warns once per unknown
    X = enc.transform(valid[["main_category"]]).toarray()
print(f"handle_unknown='ignore': {int((X.sum(axis=1) == 0).sum())}"
      f" validation rows become all zeros")

# As a share of spend, the column exists from the start, but each
# model can learn its weight only from the outcomes known at its mark.
history = table[table.end_date >= HISTORY_FROM]
print(f"\n{'Validation cohort':<19}{'known rows':>11}{'buying it':>11}"
      f"{'left':>6}{'weight':>9}")
for mark, cohort in valid.groupby("moment"):
    rows = known_by(history, mark)
    buying = rows[rows.share_sanitation > 0]
    w = FeaturedLasso(["share_sanitation"]).fit(rows).weights()
    print(f"  mark {mark:%Y-%m-%d}{len(rows):>13,}{len(buying):>11,}"
          f"{int(buying.not_renewed.sum()):>6}"
          f"{w.share_sanitation:>+9.3f}")
