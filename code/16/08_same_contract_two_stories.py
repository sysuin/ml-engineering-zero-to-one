# What an explanation describes: the same 240 calls explained by two
# models that rank about as well, and October's list explained by the
# lasso fitted in May and the lasso fitted in October.
import warnings

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import by_capacity
from foresight.evaluate import SPLITS
from foresight.explain import (TOGETHER, by_cohort, contributions,
                               scorer)
from foresight.models.boosting import RenewalBooster
from foresight.train import make_model

warnings.filterwarnings("ignore", message="LightGBM binary classifier")
table = pd.read_parquet(TABLE)
val = SPLITS["validation"]
rows = table[table.end_date.between(*val)].reset_index(drop=True)
lasso = by_cohort(table, *val, make_model())
booster = by_cohort(table, *val, RenewalBooster)
called = rows[by_capacity(rows.assign(model=scorer(lasso)(rows)))]
SHORT = {"days_since_order": "gap", "discount_pct": "discount",
         "spend_365": "spend", "tenure_days": "tenure"}


def top(models, part_of):
    """Each contract's largest part, the order counts taken as one."""
    out = []
    for mark, c in part_of.groupby("moment"):
        _, parts = contributions(models[mark], c)
        for name, cols in TOGETHER.items():
            parts[name] = parts[list(cols)].sum(axis=1)
            parts = parts.drop(columns=list(cols))
        out.append(parts.idxmax(axis=1).replace(SHORT))
    return pd.concat(out).reindex(part_of.index)


a, b = top(lasso, called), top(booster, called)
print(f"The {len(called)} calls on the validation lists: largest part")
both = pd.crosstab(a.rename("lasso"), b.rename("booster"))
print(both.to_string())
print(f"Same largest part under both models: {(a == b).mean():.1%}")

october = called[called.moment == called.moment.max()]
may = {m: lasso[min(lasso)] for m in lasso}          # May's fit, always
x, y = top(lasso, october), top(may, october)
print(f"\nOctober's {len(october)} calls, the lasso fitted in October"
      f" against May")
print(pd.crosstab(x.rename("October"), y.rename("May")).to_string())
print(f"Same largest part: {(x == y).mean():.1%}")
