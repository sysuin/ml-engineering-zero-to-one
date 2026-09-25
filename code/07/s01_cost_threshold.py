# Exercise 1: the threshold that makes the most money on validation.
import numpy as np

from foresight.costs import CostMatrix
from foresight.models.logistic import (TRAIN, VALIDATION, RenewalRisk,
                                       load)

train, valid = load(*TRAIN), load(*VALIDATION)
p = RenewalRisk().fit(train).predict_proba(valid)
y = valid.not_renewed.to_numpy()
months = valid.moment.nunique()
costs = CostMatrix(value_at_stake=2854, save_rate=0.25,
                   call_hours=1.5, hour_cost=60)


def net(t):
    called = p >= t
    tp = int((called & (y == 1)).sum())
    return costs.net_value(tp, int(called.sum()) - tp), called.sum()


grid = np.round(np.arange(0.02, 0.50, 0.001), 3)
values = [net(t)[0] for t in grid]
best = grid[int(np.argmax(values))]
print(f"  {'threshold':>9}{'calls a month':>15}{'net $':>10}")
for name, t in (("best", best), ("break-even", costs.break_even())):
    value, calls = net(t)
    print(f"  {t:>9.1%}{calls / months:>15.0f}{value:>+10,.0f}  {name}")
near = [t for t, v in zip(grid, values) if v >= max(values) - 1_000]
print(f"Within $1,000 of the best: {min(near):.1%} to {max(near):.1%}")
