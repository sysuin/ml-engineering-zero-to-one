# Cut v0.4's validation scores at one threshold after another: what
# each list made, and what the model's own chances said it would make.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import (COSTS, cost_of_errors,
                              expected_cost_of_errors, expected_value,
                              net_value)
from foresight.evaluate import SPLITS, backtest
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
s = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
months = s.moment.nunique()
even = COSTS.break_even()


def row(t):
    called = p > t
    return {"threshold": t, "calls": int(called.sum()),
            "leavers": int(called[y == 1].sum()),
            "net": net_value(y, called),
            "expected": expected_value(p, called),
            "cost": cost_of_errors(y, called),
            "expected cost": expected_cost_of_errors(p, called)}


sweep = [row(t) for t in np.round(np.arange(0.02, 0.401, 0.0025), 4)]
best = max(sweep, key=lambda r: r["net"])
print(f"Break-even: {even:.1%}. Validation, {months} cohorts\n")
print(f"{'threshold':>10}{'calls/month':>13}{'leavers':>9}"
      f"{'made $':>10}{'model said':>12}")
for r in [row(t) for t in (0.05, 0.08, 0.10)] + [row(even)] + [
        row(t) for t in (0.15, 0.20, 0.30)] + [best]:
    mark = {even: " <- break-even",
            best["threshold"]: " <- best here"}.get(r["threshold"], "")
    print(f"{r['threshold']:>10.1%}{r['calls'] / months:>13.1f}"
          f"{r['leavers']:>9}{r['net']:>+10,.0f}"
          f"{r['expected']:>+12,.0f}{mark}")
gap = row(even)["net"] - best["net"]
print(f"\nThe break-even list against the best one here:"
      f" {gap:+,.0f} ({gap / best['net']:+.1%})")
steady = [r["threshold"] for r in sweep if r["net"] > 17_000]
print(f"Every threshold that made over $17,000 lies between"
      f" {min(steady):.1%} and {max(steady):.1%}")

with open("code/14/04_threshold_by_cost.json", "w") as f:
    json.dump({"break_even": even, "best": best["threshold"],
               "sweep": sweep, "at_break_even": row(even)}, f)
