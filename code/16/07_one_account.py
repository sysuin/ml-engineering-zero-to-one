# One account on October's list, read column by column: its value, the
# typical value, and the part each model gives it, in log-odds.
import warnings

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import HISTORY_FROM, known_by
from foresight.explain import COLUMNS, contributions, facts
from foresight.models.boosting import RenewalBooster
from foresight.score import model

warnings.filterwarnings("ignore", message="LightGBM binary classifier")
table = pd.read_parquet(TABLE)
row = table[table.contract_id == 10222]          # Wickholm Foods
known = known_by(table[table.end_date >= HISTORY_FROM],
                 row.moment.iloc[0])
v06 = model().fit(known)                         # lasso, calibrated
booster = RenewalBooster().fit(known)
typical = facts(known)

b6, p6 = contributions(v06, row)
bb, pb = contributions(booster, row)
print(f"Contract 10222, Wickholm Foods, marked"
      f" {row.moment.iloc[0]:%Y-%m-%d}")
print(f"{'column':<18}{'value':>15}{'typical':>10}{'v0.6':>8}"
      f"{'booster':>9}")
for c in sorted(COLUMNS, key=lambda c: -p6[c].iloc[0]):
    v = row[c].iloc[0]
    t = typical.get(c, "")
    t = "" if isinstance(t, dict) else t
    v = f"{v:,.0f}" if isinstance(v, float) else str(v)
    t = f"{t:,.0f}" if t != "" else ""
    print(f"{c:<18}{v:>15}{t:>10}{p6[c].iloc[0]:>+8.2f}"
          f"{pb[c].iloc[0]:>+9.2f}")
for name, base, parts, m in (("v0.6", b6, p6, v06),
                             ("booster", bb, pb, booster)):
    z = base[0] + parts.iloc[0].sum()
    print(f"{name:<8} base {base[0]:+.2f} + parts"
          f" {parts.iloc[0].sum():+.2f} = {z:+.2f},"
          f" a chance of {1 / (1 + np.exp(-z)):.1%}")
print(f"Platt map: log-odds x {v06.map_.a_:.3f} {v06.map_.b_:+.3f}")
print(f"Outcome: {'left' if row.not_renewed.iloc[0] else 'renewed'}")
