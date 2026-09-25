# Steps before the model, fitted on training and validation rows
# together: a scaler, a fill for gaps, and a choice of columns. Fitted
# on the training split, scored on validation, as Chapter 7 did.
import numpy as np
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc
from foresight.models.featured import FeaturedLasso
from foresight.models.linear import Standardiser
from foresight.models.logistic import at_capacity

table = pd.read_parquet(TABLE)
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
valid = table[table.end_date.between(*SPLITS["validation"])]
both = pd.concat([train, valid])


def result(model, rows=valid):
    p = model.predict_proba(rows)
    hits = at_capacity(rows, p)["leavers"]
    return f"{hits:>7}{auc(rows.not_renewed, p):>9.4f}"


print(f"{'Fitted on training, scored on validation':<46}"
      "leavers    AUC")
honest = FeaturedLasso().fit(train)
print(f"  {'v0.4, every step on the training rows':<44}"
      f"{result(honest)}")

# 1. The scaler's means and deviations from both periods.
leaky = FeaturedLasso().fit(train)
scale = Standardiser().fit(leaky.columns(both).to_numpy())
leaky.scaler_ = scale                   # refit with the shared scale
leaky.model_ = leaky.model_.fit(
    scale.transform(leaky.columns(train).to_numpy()), train.not_renewed)
print(f"  {'the scaler fitted on both periods':<44}{result(leaky)}")

# 2. The missing discounts filled with an average, from both periods
#    or from training alone (Chapter 5: they are the legacy terms).
for name, rows in (("both periods", both), ("training", train)):
    mean = rows.discount_pct.astype(float).mean()
    fit, score = (t.assign(discount_pct=t.discount_pct.astype(float)
                           .fillna(mean)) for t in (train, valid))
    print(f"  {'gaps filled with the mean of ' + name:<44}"
          f"{result(FeaturedLasso().fit(fit), score)}"
          f"\n  {'':4}the mean discount: {mean:.2f}%")

# 3. Twenty columns chosen from 500 of pure noise, by how well they
#    track the label: once with validation's labels, once without.
g = rng()
noise = pd.DataFrame(g.normal(size=(len(both), 500)), index=both.index,
                     columns=[f"noise_{j}" for j in range(500)])
both = both.join(noise)
tr, va = both.loc[train.index], both.loc[valid.index]
for name, rows in (("chosen with both periods' labels", both),
                   ("chosen with the training labels", tr)):
    r = rows[noise.columns].corrwith(rows.not_renewed).abs()
    pick = list(r.nlargest(20).index)
    sign = np.sign(both[pick].corrwith(both.not_renewed))
    alone = auc(va.not_renewed, va[pick] @ sign)
    model = FeaturedLasso(pick).fit(tr)
    print(f"  {'noise ' + name:<44}{result(model, va)}"
          f"\n  {'':4}the twenty noise columns alone,"
          f" validation AUC {alone:.3f}")
