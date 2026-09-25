# Features from the order lines: what an account buys, and whose
# products they are. Rates on the training rows, with rough margins.
import numpy as np
import pandas as pd

from foresight.evaluate import auc
from foresight.features.build import load
from foresight.features.registry import names
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


rates(train.categories_365.clip(lower=2).map(
          lambda n: f"{n:.0f} categories" if n > 2 else "2 or fewer"),
      "Training, by breadth")
rates(pd.cut(train.supplier_voss, [-1, 0, 0.1, 0.25, 1],
             labels=["no Voss", "up to 10%", "10% to 25%",
                     "over 25%"]),
      "Training, by Voss share")

print(f"\n{'One column alone, training':<30}{'AUC':>6}")
for c in names(("products", "suppliers")):
    a = auc(y, train[c])
    print(f"  {c:<28}{max(a, 1 - a):>6.3f}")
