# Nested cross-validation in time. Thirty random settings are scored on
# every tuning cohort once. Each of the last four cohorts is then
# predicted by the setting chosen on the cohorts known at its mark, and
# the four together by the setting that looks best on them.
# timeout: 300
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import auc, hits_at_k
from foresight.models.logistic import log_loss
from foresight.tune import draw, score

table = pd.read_parquet(TABLE)
settings = draw(30)
scored = [score(table, s)["scored"] for s in settings]
incumbent = score(table)["scored"]              # Chapter 11's setting
ended = incumbent.groupby("moment").end_date.max()     # mark -> end
marks = list(ended.index)
outer = marks[-4:]


def measure(s, cohorts):
    """Rows of these cohorts: log loss, AUC, leavers in the top 40s."""
    s = s[s.moment.isin(cohorts)]
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    hits = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in s.groupby("moment"))
    return log_loss(y, p), auc(y, p), hits


print("Gain over Chapter 11's setting, in log loss")
print(f"{'outer cohort':<14}{'known before it':<17}{'chosen':>6}"
      f"{'claimed':>10}{'real':>10}")
honest, folds = [], []
for mark in outer:
    inner = [m for m in marks if ended[m] < mark]   # known at the mark
    losses = [measure(s, inner)[0] for s in scored]
    i = int(np.argmin(losses))
    honest.append(scored[i][scored[i].moment == mark])
    claimed = measure(incumbent, inner)[0] - losses[i]
    real = (measure(incumbent, [mark])[0]
            - measure(scored[i], [mark])[0])
    names = ", ".join(f"{ended[m]:%b}" for m in inner)
    print(f"{ended[mark]:%Y-%m-%d}{'':4}{names:<17}{i + 1:>6}"
          f"{claimed:>+10.5f}{real:>+10.5f}")
    folds.append({"outer": f"{ended[mark]:%Y-%m}",
                  "inner": [f"{ended[m]:%Y-%m}" for m in inner]})

h = pd.concat(honest)
lines = {"Chapter 11's setting": measure(incumbent, outer),
         "chosen on earlier cohorts": measure(h, outer),
         "chosen on the four cohorts": min(measure(s, outer)
                                           for s in scored)}
print(f"\nThe four outer cohorts together: {len(h)} contracts,"
      f" {int(h.not_renewed.sum())} leavers\n")
print(f"{'':30}{'log loss':>9}{'AUC':>7}{'leavers':>9}")
for name, (ll, a, hits) in lines.items():
    print(f"{name:<30}{ll:>9.5f}{a:>7.3f}{hits:>9}")
optimism = (lines["chosen on earlier cohorts"][0]
            - lines["chosen on the four cohorts"][0])
print(f"\nOptimism of choosing on the rows you report: {optimism:.5f}")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    json.dump({"folds": folds,
               "cohorts": [f"{ended[m]:%Y-%m}" for m in marks],
               "lines": {k: v[0] for k, v in lines.items()}}, f)
