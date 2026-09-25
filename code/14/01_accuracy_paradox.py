# Six lists for the validation cohorts, judged by accuracy and by money.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, net_value
from foresight.evaluate import SPLITS, backtest
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
train = table[table.end_date.between("2023-01-01", "2024-06-30")]
print(f"Training: {len(train):,} renewals,"
      f" {train.not_renewed.mean():.1%} left;"
      f" 'nobody leaves' is right {1 - train.not_renewed.mean():.1%}")

s = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
y = s.not_renewed.to_numpy()
print(f"Validation: {len(s):,} renewals, {y.sum()} left"
      f" ({y.mean():.1%}), scored by v0.4's lasso\n")
lists = {
    "nobody leaves": np.zeros(len(s), bool),
    "everybody leaves": np.ones(len(s), bool),
    "lasso above 50%": s.model.to_numpy() > 0.5,
    "rule, top 40": by_capacity(s, score="rule"),
    "lasso, top 40": by_capacity(s),
    "perfect, top 40": by_capacity(s.assign(p=y), score="p"),
}
print(f"{'list':<18}{'accuracy':>9}{'calls':>7}{'leavers':>9}"
      f"{'net $':>10}")
rows = {}
for name, called in lists.items():
    acc = float((called == (y == 1)).mean())
    net = net_value(y, called)
    rows[name] = {"accuracy": acc, "calls": int(called.sum()),
                  "leavers": int(called[y == 1].sum()), "net": net}
    print(f"{name:<18}{acc:>9.1%}{called.sum():>7,}"
          f"{called[y == 1].sum():>9}{net:>+10,.0f}")

with open("code/14/01_accuracy_paradox.json", "w") as f:
    json.dump(rows, f, indent=1)
