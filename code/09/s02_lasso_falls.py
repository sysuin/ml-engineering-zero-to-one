# Exercise 2: where does the lasso start to cost leavers?
import textwrap

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import backtest, hits_at_k
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
