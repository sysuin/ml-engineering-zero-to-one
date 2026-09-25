# Platt scaling and isotonic regression, each learned on the three
# latest cohorts a model may see and applied to the cohort it scores.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import (by_cost, calibrated, calibration_slope,
                              net_value, reliability)
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.boosting import RenewalBooster
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
models = {"lasso": maker("l1", 0.002), "booster": RenewalBooster}
print(f"{'':<18}{'leavers':>8}{'log loss':>10}{'slope':>7}"
      f"{'mean p':>8}{'calls':>7}{'made $':>9}")
saved = {}
for name, make in models.items():
    for method in ("as fitted", "platt", "isotonic"):
        f = make if method == "as fitted" else calibrated(make, method)
        s = backtest(table, *SPLITS["validation"], f)
        y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
        m, called = measure(s), by_cost(p)
        label = f"{name}, {method}" if method == "as fitted" \
            else f"  {method}"
        print(f"{label:<18}{m['precision']['model'] * 240:>8.0f}"
              f"{m['log loss']['model']:>10.4f}"
              f"{calibration_slope(p, y)[0]:>7.2f}{p.mean():>8.1%}"
              f"{called.sum():>7}{net_value(y, called):>+9,.0f}")
        saved[f"{name} {method}"] = reliability(p, y).to_dict("list")
        if method == "isotonic":
            print(f"    {(p == 0).sum()} contracts scored exactly 0,"
                  f" of which {int(y[p == 0].sum())} left")
print("calls, made $: every contract above the 12.6% break-even")

with open("code/14/09_calibrate.json", "w") as f:
    json.dump(saved, f)
