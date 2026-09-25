# Target encoding the postcode area, twice: with each row's own label
# inside its encoding, and out of fold. Fitted on the training split,
# scored on validation, as Chapter 7 did.
import json

from foresight.evaluate import SPLITS, auc
from foresight.features.build import load
from foresight.features.encoding import encode, out_of_fold
from foresight.features.sources import load_sources
from foresight.models.featured import TRAINING, FeaturedLasso
from foresight.models.logistic import at_capacity

table = load()
postcode = load_sources().accounts.set_index("account_id").postcode
table["area"] = table.account_id.map(postcode).str[:3]
train = table[table.end_date.between(*TRAINING)].copy()
valid = table[table.end_date.between(*SPLITS["validation"])].copy()

# Validation rows are always encoded from all the training rows.
valid["area_rate"] = encode(train.area, train.not_renewed, valid.area)
ways = {
    "naive": encode(train.area, train.not_renewed, train.area),
    "out of fold": out_of_fold(train.area, train.not_renewed,
                               train.account_id)}

print(f"{'':16}{'encoding alone, AUC':>21}{'lasso with it, AUC':>21}")
print(f"{'':16}{'training':>11}{'validation':>11}"
      f"{'training':>10}{'validation':>11}{'leavers':>9}")
out = {}
for name, rates in [("without it", None), *ways.items()]:
    if rates is None:
        model = FeaturedLasso().fit(train)
        enc = ("", "")
    else:
        train["area_rate"] = rates
        model = FeaturedLasso(["area_rate"]).fit(train)
        enc = (f"{auc(train.not_renewed, rates):.3f}",
               f"{auc(valid.not_renewed, valid.area_rate):.3f}")
    fit_auc = auc(train.not_renewed, model.predict_proba(train))
    p = model.predict_proba(valid)
    val_auc = auc(valid.not_renewed, p)
    hits = at_capacity(valid, p)["leavers"]
    print(f"  {name:<14}{enc[0]:>11}{enc[1]:>11}{fit_auc:>10.3f}"
          f"{val_auc:>11.3f}{hits:>9}")
    out[name] = {"train": fit_auc, "valid": val_auc, "hits": hits}

print(f"\nAreas: {train.area.nunique()} in training, median"
      f" {train.area.value_counts().median():.0f} rows each;"
      f" {valid.area.isin(train.area).mean():.1%}\nof validation rows"
      f" are in an area seen in training")
with open("code/12/07_target_encoding_leak.json", "w") as f:
    json.dump(out, f)
