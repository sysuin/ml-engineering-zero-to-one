# The list at every capacity, and the probabilities judged as numbers.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, backtest, brier, measure
from foresight.models.logistic import log_loss

table = pd.read_parquet(TABLE)
scored = backtest(table, *SPLITS["validation"])

print("Validation cohorts: leavers reached and precision, by calls")
print(f"{'calls a cohort':<16}{'model':>12}{'rule':>12}{'random':>12}"
      f"{'perfect':>12}")
curve = {}
for k in range(5, 151, 5):
    m = measure(scored, k)
    curve[k] = {n: m["precision"][n] for n in m["precision"]}
    if k in (10, 20, 40, 80, 120):
        calls = sum(min(k, n) for n in scored.groupby("moment").size())
        cells = [f"{round(v * calls):>5} {v:>5.1%}"
                 for v in curve[k].values()]
        print(f"{k:<16}" + " ".join(f"{c:>11}" for c in cells))

y, p = scored.not_renewed.to_numpy(), scored.model.to_numpy()
base = scored.base.to_numpy()          # training rate at each mark
print("\nThe probabilities themselves")
print(f"  {'':24}{'log loss':>9}{'Brier':>8}")
print(f"  {'model':24}{log_loss(y, p):>9.4f}{brier(y, p):>8.4f}")
print(f"  {'base rate for everyone':24}{log_loss(y, base):>9.4f}"
      f"{brier(y, base):>8.4f}")
print(f"  mean predicted {p.mean():.1%},"
      f" share that left {y.mean():.1%}")
skill = 1 - brier(y, p) / brier(y, base)
print(f"  Brier skill against the base rate: {skill:.1%}")

with open("code/08/10_precision_at_k.json", "w") as f:
    json.dump({"curve": curve}, f)
