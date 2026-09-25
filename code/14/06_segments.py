# One list, four segments: how often each was right and wrong, and what
# each segment's own break-even would be.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE, rng
from foresight.costs import CostMatrix
from foresight.data.build_table import TABLE
from foresight.decide import COSTS, by_capacity
from foresight.evaluate import SPLITS, backtest, interval, resample
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
s = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
s["called"] = by_capacity(s)
g = rng()
draws = [resample(s, g).groupby("segment").not_renewed.mean()
         for _ in range(2000)]
draws = pd.DataFrame(draws)
print(f"{'segment':<15}{'contracts':>9}{'left':>6}{'95% range':>12}"
      f"{'model':>7}{'calls':>6}{'recall':>7}{'FPR':>6}")
for seg, c in s.groupby("segment"):
    lo, hi = interval(draws[seg])
    left, stayed = c.not_renewed == 1, c.not_renewed == 0
    print(f"{seg:<15}{len(c):>9,}{left.mean():>6.1%}"
          f"{lo:>6.1%}-{hi:<5.1%}{c.model.mean():>7.1%}"
          f"{c.called.sum():>6}{c.called[left].mean():>7.0%}"
          f"{c.called[stayed].mean():>6.1%}")
print("model: average chance given; FPR: share of renewers called")

# Chapter 3's value at stake, a year of gross profit, by segment: the
# long tail's training leavers, as the brief measured it.
profit = pd.read_sql_query("""
    SELECT c.contract_id,
           COALESCE(SUM(l.qty * (l.unit_price - l.unit_cost)), 0) AS gp
    FROM contracts c
    JOIN accounts a USING (account_id)
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date <  date(c.end_date, '-90 days')
          AND o.order_date >= date(c.end_date, '-455 days')
    LEFT JOIN order_lines l USING (order_id)
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome = 'not_renewed' AND a.is_key_account = 0
    GROUP BY c.contract_id""", sqlite3.connect(ML_WAREHOUSE))
lost = profit.merge(table[["contract_id", "segment"]])
print(f"\n{'segment':<15}{'leavers':>8}{'value':>8}{'break-even':>11}"
      f"{'calls':>7}{'leavers':>8}{'made $':>9}")
for seg, v in lost.groupby("segment").gp:
    own = CostMatrix(round(v.mean()), COSTS.save_rate,
                     COSTS.call_hours, COSTS.hour_cost)
    c = s[s.segment == seg]
    called = c.model.to_numpy() > own.break_even()
    y = c.not_renewed.to_numpy()
    made = own.net_value(called[y == 1].sum(), called[y == 0].sum())
    print(f"{seg:<15}{len(v):>8}{own.value_at_stake:>8,}"
          f"{own.break_even():>11.1%}{called.sum():>7}"
          f"{called[y == 1].sum():>8}{made:>+9,.0f}")
print("calls, leavers, made $: validation, at each segment's own"
      " cut")
