# Forty calls a cohort against the break-even, cohort by cohort, and
# what each way of cutting the list made on validation.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import (COSTS, by_capacity, by_cost,
                              expected_value, net_value)
from foresight.evaluate import SPLITS, backtest
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
s = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
s["top"], s["even"] = by_capacity(s), by_cost(s.model)
print(f"{'cohort mark':<13}{'contracts':>10}{'above':>7}"
      f"{'40th score':>12}{'leavers in top 40':>19}")
cohorts = []
for mark, c in s.groupby("moment"):
    fortieth = c.model[c.top].min()
    cohorts.append({"mark": f"{mark:%Y-%m-%d}", "contracts": len(c),
                    "above": int(c.even.sum()), "fortieth": fortieth,
                    "leavers": int(c.not_renewed.sum())})
    print(f"{mark:%Y-%m-%d}{len(c):>13}{c.even.sum():>7}"
          f"{fortieth:>12.1%}{c.not_renewed[c.top].sum():>19}")
print(f"above: contracts above the {COSTS.break_even():.1%} break-even")

y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
top, even = s.top.to_numpy(), s.even.to_numpy()
lists = {"top 40": top, "above break-even": even,
         "top 40, not above": top & ~even,
         "above, not top 40": even & ~top}
print(f"\n{'list':<18}{'calls':>6}{'leavers':>9}{'made $':>10}"
      f"{'model said $':>14}")
for name, called in lists.items():
    print(f"{name:<18}{called.sum():>6}{called[y == 1].sum():>9}"
          f"{net_value(y, called):>+10,.0f}"
          f"{expected_value(p, called):>+14,.0f}")

with open("code/14/05_threshold_by_capacity.json", "w") as f:
    json.dump({"break_even": COSTS.break_even(), "cohorts": cohorts},
              f)
