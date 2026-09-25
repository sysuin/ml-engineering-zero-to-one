# Halloway at its mark, 1 April 2024: what each model said, where that
# put it among the 338 contracts, and why, part by part.
import json
import warnings

import numpy as np
import pandas as pd

from foresight.evaluate import HISTORY_FROM, known_by
from foresight.explain import COLUMNS, contributions, unfamiliar
from foresight.features.build import load
from foresight.features.registry import names
from foresight.models.boosting import RenewalBooster
from foresight.models.featured import FeaturedBooster, FeaturedLasso
from foresight.score import model
from foresight.train import make_model

warnings.filterwarnings("ignore", message="LightGBM binary classifier")
table = load()
cohort = table[table.moment == "2024-04-01"]
known = known_by(table[table.end_date >= HISTORY_FROM], "2024-04-01")
h = cohort[cohort.contract_id == 8118]
tickets = names("tickets")
models = {"v0.6: lasso, calibrated": model(),
          "v0.5: lasso": make_model()(),
          "booster": RenewalBooster(),
          "lasso + ticket features": FeaturedLasso(tickets),
          "booster + ticket features": FeaturedBooster(tickets)}
print(f"{'':27}{'chance':>12}{'rank':>6}{'40th chance':>13}")
cuts = {}
for name, m in models.items():
    p = pd.Series(m.fit(known).predict_proba(cohort),
                  index=cohort.contract_id)
    rank = int((p >= p[8118]).sum())     # ties counted against it
    cut = p.sort_values(ascending=False).iloc[39]
    cuts[name] = float(cut)
    print(f"  {name:<25}{p[8118]:>12.2g}{rank:>6}{cut:>13.1%}")

lasso, booster = models["v0.5: lasso"], models["booster"]
lb, lp = contributions(lasso, h)
bb, bp = contributions(booster, h)
print(f"\n{'Parts, in log-odds':<40}{'lasso':>8}{'booster':>10}")
for c in sorted(COLUMNS, key=lambda c: lp[c].iloc[0]):
    print(f"  {c:<38}{lp[c].iloc[0]:>+8.2f}{bp[c].iloc[0]:>+10.2f}")
print(f"  {'base':<38}{lb[0]:>+8.2f}{bb[0]:>+10.2f}")
for name, b, p in (("lasso", lb, lp), ("booster", bb, bp)):
    z = b[0] + p.iloc[0].sum()
    print(f"  {name} log-odds {z:+.2f}: a chance of"
          f" {1 / (1 + np.exp(-z)):.2g}")
x = lasso.scaler_.transform(lasso.columns(h).to_numpy(float))[0]
z = dict(zip(lasso.columns_, x))
print(f"orders_90d stands {z['orders_90d']:.1f} standard deviations"
      f" above the average")
odd = unfamiliar(known, h).iloc[0]
print("Rarely seen, fewer than 20 training rows as far out:\n  "
      + ", ".join(odd.index[odd]))

with open("code/16/12_halloway_scores.json", "w") as f:
    json.dump({m: {"base": float(b[0]), "cut": cuts[k],
                   "parts": {c: round(float(p[c].iloc[0]), 4)
                             for c in COLUMNS}}
               for m, k, b, p in (("lasso", "v0.5: lasso", lb, lp),
                                  ("booster", "booster", bb, bp))}, f)
