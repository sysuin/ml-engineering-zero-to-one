# Exercise 2: where does the lasso start to cost leavers?
import textwrap

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import (HISTORY_FROM, auc, backtest, hits_at_k,
                                known_by)
from foresight.models.logistic import TRAIN
from foresight.models.regularised import (TUNE, RegularisedRisk,
                                          maker, tune)

table = pd.read_parquet(TABLE)
strengths = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
t = tune(table, penalties=("l1",), strengths=strengths)
train = table[table.end_date.between(*TRAIN)]
print("tuning cohorts; columns kept by a fit on the training split")
print(f"{'strength':>9}{'AUC':>8}{'top 40':>8}{'kept':>6}")
for _, r in t.iterrows():
    w = RegularisedRisk("l1", r.strength).fit(train).weights()
    kept = [c.replace("segment=", "") for c in w.index[w != 0]]
    print(f"{r.strength:>9g}{r.auc:>8.4f}{r.hits:>8}{len(kept):>6}")
    if len(kept) <= 6:
        for line in textwrap.wrap(", ".join(kept), 50,
                                  break_on_hyphens=False):
            print(f"{'':>12}{line}")
s = backtest(table, *TUNE, maker("l2", 0.0))
rule = sum(hits_at_k(c.not_renewed, c.rule, c.contract_id)
           for _, c in s.groupby("moment"))
print(f"the rule on the same cohorts: top 40 {rule}")

# Pooled AUC also compares contracts from different cohorts. Averaged
# cohort by cohort, it compares each contract only with its own month.
def by_cohort(s, score):
    return s.groupby("moment").apply(
        lambda c: auc(c.not_renewed, c[score])).mean()


history = table[table.end_date >= HISTORY_FROM]
print("\ncohort by cohort: mean AUC, and the columns each cohort's"
      "\nmodel keeps (fewest to most)")
for st in strengths:
    s = backtest(table, *TUNE, maker("l1", st))
    kept = [int((RegularisedRisk("l1", st).fit(known_by(history, m))
                 .weights() != 0).sum()) for m in s.moment.unique()]
    print(f"{st:>9g}{by_cohort(s, 'model'):>8.4f}"
          f"{min(kept):>6} to {max(kept)}")
print(f"{'rule':>9}{by_cohort(s, 'rule'):>8.4f}")
