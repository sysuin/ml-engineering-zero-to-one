# Exercise 2: target encoding broken on purpose, two ways. Inside the
# training rows (each row's label in its own encoding), and across the
# split (validation labels in the encoding the model is scored with).
from foresight.evaluate import SPLITS, auc
from foresight.features.build import load
from foresight.features.encoding import encode, out_of_fold
from foresight.features.sources import load_sources
from foresight.models.featured import TRAINING, FeaturedLasso
from foresight.models.logistic import at_capacity

table = load()
postcode = load_sources().accounts.set_index("account_id").postcode
table["postcode"] = table.account_id.map(postcode)
train = table[table.end_date.between(*TRAINING)].copy()
valid = table[table.end_date.between(*SPLITS["validation"])].copy()
both = table[table.end_date <= SPLITS["validation"][1]]
fair = encode(train.postcode, train.not_renewed, valid.postcode)

ways = {
    "out of fold": (out_of_fold(train.postcode, train.not_renewed,
                                train.account_id), fair),
    "own label inside": (encode(train.postcode, train.not_renewed,
                                train.postcode), fair),
    "validation inside": (encode(both.postcode, both.not_renewed,
                                 train.postcode),
                          encode(both.postcode, both.not_renewed,
                                 valid.postcode))}
print(f"{'Full postcode':<20}{'training AUC':>13}{'validation AUC':>16}"
      f"{'leavers':>9}")
for name, (tr, va) in ways.items():
    train["pc_rate"], valid["pc_rate"] = tr, va
    m = FeaturedLasso(["pc_rate"]).fit(train)
    fit, p = m.predict_proba(train), m.predict_proba(valid)
    print(f"  {name:<18}{auc(train.not_renewed, fit):>13.3f}"
          f"{auc(valid.not_renewed, p):>16.3f}"
          f"{at_capacity(valid, p)['leavers']:>9}")
