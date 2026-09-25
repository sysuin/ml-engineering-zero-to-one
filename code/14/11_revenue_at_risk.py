# Expected revenue at risk by region on the validation cohorts: each
# contract's chance of leaving times a quarter's revenue at its run
# rate, from v0.4's lasso as fitted and Platt-calibrated out of time.
import json
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE, rng
from foresight.data.build_table import TABLE
from foresight.decide import (calibrated, revenue_at_risk, simulate,
                              value_at_stake)
from foresight.evaluate import SPLITS, backtest, interval, resample
from foresight.models.boosting import RenewalBooster
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
key = pd.read_sql_query("SELECT account_id, is_key_account"
                        " FROM accounts",
                        sqlite3.connect(ML_WAREHOUSE))
raw = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
cal = backtest(table, *SPLITS["validation"],
               calibrated(maker("l1", 0.002)))
boost = backtest(table, *SPLITS["validation"],
                 calibrated(RenewalBooster))
s = raw.merge(key).assign(platt=cal.model.to_numpy(),
                          booster=boost.model.to_numpy(),
                          value=value_at_stake(raw))
tail, keys = s[s.is_key_account == 0], s[s.is_key_account == 1]

risk = {n: revenue_at_risk(tail[c], tail.value, tail.region)
        for n, c in (("lasso", "model"), ("Platt", "platt"))}
walked = revenue_at_risk(tail.not_renewed, tail.value, tail.region)
band = simulate(tail.platt, tail.value, tail.region)
band["all"] = band.sum(axis=1)
print(f"Long tail: {len(tail):,} contracts, {tail.not_renewed.sum()}"
      f" left. A quarter's revenue, $\n")
print(f"{'region':<11}{'lasso':>10}{'Platt':>10}{'95% range':>21}"
      f"{'walked':>10}")
out = {}
for r in [*walked.index, "all"]:
    a, b, w = ((x.sum() if r == "all" else x[r])
               for x in (risk["lasso"], risk["Platt"], walked))
    lo, hi = interval(band[r])
    out[r] = {"lasso": a, "platt": b, "lo": lo, "hi": hi, "walked": w}
    print(f"{r:<11}{a:>10,.0f}{b:>10,.0f}{lo:>11,.0f} to{hi:>8,.0f}"
          f"{w:>10,.0f}")
print("range: the Platt total in 95% of 2,000 simulated half-years")

g = rng()
gaps = {c: [] for c in ("model", "platt", "booster")}
for _ in range(2000):
    d = resample(tail, g)
    for c in gaps:
        gaps[c].append(float(((d[c] - d.not_renewed) * d.value).sum()))
print("\nForecast minus walked, all regions, resampled (95%)")
for c, name in (("model", "lasso as fitted"), ("platt", "lasso, Platt"),
                ("booster", "booster, Platt")):
    lo, hi = interval(gaps[c])
    gap = ((tail[c] - tail.not_renewed) * tail.value).sum()
    print(f"  {name:<16}{gap:>+10,.0f}  ({lo:+,.0f} to {hi:+,.0f})")

kb = simulate(keys.platt, keys.value, np.zeros(len(keys)))[0]
print(f"\nKey accounts: {len(keys)} contracts,"
      f" {keys.not_renewed.sum()} left")
print(f"  at risk: lasso {(keys.model * keys.value).sum():,.0f},"
      f" Platt {(keys.platt * keys.value).sum():,.0f};"
      f" 95% range {interval(kb)[0]:,.0f} to {interval(kb)[1]:,.0f}")

with open("code/14/11_revenue_at_risk.json", "w") as f:
    json.dump(out, f, indent=1)
