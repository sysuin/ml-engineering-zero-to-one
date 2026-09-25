# Each group of features on the validation backtest: what it adds to
# v0.4's lasso and to the booster, paired, against the headroom left
# below the ceiling (read from the generator for this purpose only).
# timeout: 400
import json

import pandas as pd

from foresight.config import TRUTH
from foresight.evaluate import SPLITS, backtest, hits_at_k
from foresight.features.build import load
from foresight.features.registry import groups, names
from foresight.models.featured import (SHOWN, booster, evidence, gain,
                                       lasso, shown)

table = load()
V = SPLITS["validation"]
candidates = {g: names(g) for g in groups()}
ev = evidence(table, candidates)
for model in ("lasso", "booster"):      # the groups the rule kept
    kept = ev.index[ev[f"keep {model}"]]
    candidates[f"kept, {model}"] = [f for g in kept for f in names(g)]
candidates["everything"] = names()

truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
out = {}
for model, make in (("lasso", lasso), ("booster", booster)):
    base = backtest(table, *V, make())
    best = base.assign(model=truth.p_leave.reindex(
        base.contract_id).to_numpy())
    ceiling = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
                  for _, c in best.groupby("moment"))
    now = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
              for _, c in base.groupby("moment"))
    print(f"{model}: {now} leavers of the ceiling's {ceiling};"
          f" headroom {ceiling - now}")
    print(f"  {'added':<10}{SHOWN}")
    out[model] = {"now": now, "ceiling": ceiling, "groups": {}}
    for name, extra in candidates.items():
        if name.startswith("kept") and not name.endswith(model):
            continue
        g = gain(backtest(table, *V, make(extra)), base)
        print(f"  {name.removesuffix(', ' + model):<13}{shown(g)}")
        out[model]["groups"][name] = g
    print()

with open("code/12/13_validation_gains.json", "w") as f:
    json.dump(out, f)
