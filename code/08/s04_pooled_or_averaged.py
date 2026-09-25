# Exercise 4: AUC pooled over the cohorts, or averaged across them.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, auc, backtest

table = pd.read_parquet(TABLE)
scored = backtest(table, *SPLITS["validation"])
print(f"{'cohort (mark)':<15}{'leavers':>8}{'model':>8}{'rule':>8}")
per = []
for mark, c in scored.groupby("moment"):
    y = c.not_renewed
    per.append((auc(y, c.model), auc(y, c.rule)))
    print(f"{mark:%Y-%m-%d}{c.not_renewed.sum():>13}{per[-1][0]:>8.3f}"
          f"{per[-1][1]:>8.3f}")
mean = [sum(p[i] for p in per) / len(per) for i in (0, 1)]
y = scored.not_renewed
print(f"{'average':<15}{'':>8}{mean[0]:>8.3f}{mean[1]:>8.3f}")
print(f"{'pooled':<15}{'':>8}{auc(y, scored.model):>8.3f}"
      f"{auc(y, scored.rule):>8.3f}")
