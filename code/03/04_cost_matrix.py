# What each kind of error costs, in dollars and account-manager hours.
import json
import sqlite3
from pathlib import Path

from foresight.config import ML_WAREHOUSE
from foresight.costs import CostMatrix

con = sqlite3.connect(ML_WAREHOUSE)


def usd(x: float) -> str:
    return f"${x:,.0f}"


# A renewal is worth a year of the account's gross profit: measured here
# as the profit on its orders in the 365 days before its 90-day mark.
profit = con.execute("""
    SELECT a.is_key_account,
           COALESCE(SUM(l.qty * (l.unit_price - l.unit_cost)), 0)
    FROM contracts c
    JOIN accounts a USING (account_id)
    LEFT JOIN orders o
           ON o.account_id = c.account_id
          AND o.order_date <  date(c.end_date, '-90 days')
          AND o.order_date >= date(c.end_date, '-455 days')
    LEFT JOIN order_lines l USING (order_id)
    WHERE c.end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND c.outcome = 'not_renewed'
    GROUP BY c.contract_id""").fetchall()
tail = sorted(p for key, p in profit if not key)
keys = sorted(p for key, p in profit if key)
mean, median = sum(tail) / len(tail), tail[len(tail) // 2]
print("A year's gross profit, for renewals that were lost")
for name, group in (("long-tail leavers", tail),
                    ("key-account leavers", keys)):
    print(f"  {name:20}{len(group):>4}   total {usd(sum(group)):>10}")
    if group is tail:
        print(f"  {'':24}   mean  {usd(mean):>10}"
              f"   median {usd(median)}")

# The estimates: a call saves one would-be leaver in four, and takes an
# hour and a half of an account manager's time at $60 an hour.
costs = CostMatrix(value_at_stake=round(mean), save_rate=0.25,
                   call_hours=1.5, hour_cost=60)
cells = costs.cells()
print("\nOne call decision, net dollars against making no call")
print(f"  {'':12}{'would leave':>14}{'would stay':>14}")
print(f"  {'called':12}{cells['tp']:>+14,.0f}{cells['fp']:>+14,.0f}")
print(f"  {'not called':12}{cells['fn']:>14,.0f}{cells['tn']:>14,.0f}")
err = costs.error_costs()
print(f"A false positive wastes {costs.call_hours:g} hours:"
      f" {usd(err['fp'])}")
print(f"A false negative forgoes ${err['fn']:,.0f},"
      f" {err['fn'] / err['fp']:.1f} times as much")
print(f"A call pays for itself above a {costs.break_even():.1%} chance"
      " of leaving")

# What a month of 40 calls is worth, at each list's precision.
lists = json.loads(Path("code/03/03_baseline_rule.json").read_text())
calls = lists["calls_per_cohort"]
print(f"\nA month of {calls} calls"
      f" ({calls * costs.call_hours:g} hours)")
print(f"  {'':16}{'precision':>10}{'per call':>10}{'per month':>11}")
for name, precision in (("random list", lists["base_rate"]),
                        ("days-since rule", lists["rule_precision"])):
    each = costs.value_per_call(precision)
    print(f"  {name:16}{precision:>10.1%}{each:>+10,.0f}"
          f"{each * calls:>+11,.0f}")

Path("code/03/04_cost_matrix.json").write_text(json.dumps({
    **cells, "break_even": costs.break_even(),
    "value_at_stake": costs.value_at_stake,
    "save_rate": costs.save_rate,
    "call_hours": costs.call_hours, "hour_cost": costs.hour_cost},
    indent=1))
