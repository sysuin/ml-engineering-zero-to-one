# Exercise 2: twenty calls a cohort instead of forty, v0.4's lasso as
# fitted and Platt-calibrated, against the break-even.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import (by_capacity, by_cost, calibrated,
                              expected_value, net_value)
from foresight.evaluate import SPLITS, backtest
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
for name, make in (("as fitted", maker("l1", 0.002)),
                   ("Platt", calibrated(maker("l1", 0.002)))):
    s = backtest(table, *SPLITS["validation"], make)
    y, p = s.not_renewed.to_numpy(), s.model.to_numpy()
    print(f"v0.4's lasso, {name}")
    print(f"  {'list':<20}{'calls':>6}{'leavers':>9}{'made $':>10}"
          f"{'model said':>12}")
    for label, called in (("top 20", by_capacity(s, 20)),
                          ("top 40", by_capacity(s, 40)),
                          ("above break-even", by_cost(p))):
        print(f"  {label:<20}{called.sum():>6}{called[y == 1].sum():>9}"
              f"{net_value(y, called):>+10,.0f}"
              f"{expected_value(p, called):>+12,.0f}")
    over = [int(by_cost(c.model).sum())
            for _, c in s.groupby("moment")]
    cut = [c.model.nlargest(20).min() for _, c in s.groupby("moment")]
    print(f"  above break-even, by cohort: {over}")
    print("  20th score, by cohort: "
          + ", ".join(f"{v:.1%}" for v in cut))
