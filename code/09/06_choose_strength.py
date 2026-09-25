# Choosing the strength on training cohorts, by a rolling backtest.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, backtest, known_by
from foresight.models.logistic import log_loss
from foresight.models.regularised import (TUNE, RegularisedRisk,
                                          choose, maker, tune)

table = pd.read_parquet(TABLE)
results = tune(table)                   # 9 strengths x 2 penalties
cohorts = table[table.end_date.between(*TUNE)]
print(f"tuning cohorts: contracts ending {TUNE[0]} to {TUNE[1]}")
n, left = len(cohorts), cohorts.not_renewed.sum()
print(f"  {cohorts.moment.nunique()} cohorts, {n:,} contracts,"
      f" {left} leavers")
print(f"\n{'':>9}{'ridge (L2)':^27}{'lasso (L1)':^27}".rstrip())
print(f"{'strength':>9}"
      + f"{'log loss':>10}{'AUC':>8}{'top 40':>9}" * 2)
wide = results.pivot(index="strength", columns="penalty")
for s, r in wide.iterrows():
    print(f"{s:>9g}" + "".join(
        f"{r[('log loss', p)]:>10.4f}{r[('auc', p)]:>8.4f}"
        f"{int(r[('hits', p)]):>9}" for p in ("l2", "l1")))
best = choose(results)
print(f"lowest log loss: {best['penalty']}, strength"
      f" {best['strength']:g}")

print("\nlog loss by cohort, unpenalised against the choice")
free = backtest(table, *TUNE, maker("l2", 0.0))
chosen = backtest(table, *TUNE, maker(**best))
for (mark, a), (_, b) in zip(free.groupby("moment"),
                             chosen.groupby("moment")):
    y = a.not_renewed.to_numpy()
    print(f"  {mark:%Y-%m-%d}  fitted on {a.trained_on.iloc[0]:>5,}"
          f"  {log_loss(y, a.model.to_numpy()):.4f}"
          f"  {log_loss(y, b.model.to_numpy()):.4f}")

first = free.moment.min()               # the worst cohort, looked at
seen = known_by(table[table.end_date >= HISTORY_FROM], first)
public = seen[seen.segment == "Public sector"]
print(f"\nits {len(seen)} rows: {len(public)} public-sector contracts,"
      f" {public.not_renewed.sum()} left")
print(f"  {'':<14}{'weight':>7}   chances given to the two"
      " who left")
for name, model in (("unpenalised", RegularisedRisk("l2", 0.0)),
                    ("chosen", RegularisedRisk(**best))):
    w = model.fit(seen).weights()["segment=Public sector"]
    c = free[free.moment == first]
    left = c[(c.segment == "Public sector") & (c.not_renewed == 1)]
    p = model.predict_proba(left)
    print(f"  {name:<14}{w:>+7.2f}   "
          + "   ".join(f"{v:.1e}" for v in p))
