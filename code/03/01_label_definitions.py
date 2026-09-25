# Three definitions of "churned", applied to the same accounts.
import json
import sqlite3
from pathlib import Path

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)


def ids(sql: str) -> set[int]:
    return {row[0] for row in con.execute(sql)}


# Who could churn at all: a customer on 1 January 2024, with an order in
# 2023 or a contract running that day, and not already closed.
customers = ids("""
    SELECT account_id FROM accounts
    WHERE COALESCE(closed_on, '9999-12-31') >= '2024-01-01'
      AND (account_id IN (
               SELECT account_id FROM orders
               WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31')
        OR account_id IN (
               SELECT account_id FROM contracts
               WHERE start_date <= '2024-01-01'
                 AND end_date >= '2024-01-01'))""")

# A: a contract ended in the half-year and was not renewed.
not_renewed = ids("""
    SELECT account_id FROM contracts
    WHERE end_date BETWEEN '2024-01-01' AND '2024-06-30'
      AND outcome = 'not_renewed'""") & customers

# B: no order in the last 90 days of the half-year.
no_orders = customers - ids("""
    SELECT account_id FROM orders
    WHERE order_date BETWEEN '2024-04-01' AND '2024-06-30'""")

# C: second-quarter revenue less than half of first-quarter revenue.
halved = ids("""
    SELECT o.account_id FROM orders o JOIN order_lines l USING(order_id)
    WHERE o.order_date BETWEEN '2024-01-01' AND '2024-06-30'
    GROUP BY o.account_id
    HAVING SUM(CASE WHEN o.order_date < '2024-04-01'
                    THEN l.qty * l.unit_price ELSE 0 END) > 0
       AND SUM(CASE WHEN o.order_date >= '2024-04-01'
                    THEN l.qty * l.unit_price ELSE 0 END)
         < 0.5 * SUM(CASE WHEN o.order_date < '2024-04-01'
                          THEN l.qty * l.unit_price ELSE 0 END)
    """) & customers

print(f"Customers on 1 January 2024: {len(customers):,}\n")
print("Churned in the first half of 2024, by definition")
defs = {"A": ("contract not renewed", not_renewed),
        "B": ("no order for 90 days", no_orders),
        "C": ("revenue halved, Q2 on Q1", halved)}
for key, (name, flagged) in defs.items():
    print(f"  {key}  {name:<28}{len(flagged):>7,}")

print("\nHow far they agree (flagged by both, of flagged by either)")
for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
    both, either = defs[x][1] & defs[y][1], defs[x][1] | defs[y][1]
    print(f"  {x} and {y}  {len(both):>5,} of {len(either):>5,}"
          f"  {len(both) / len(either):>6.1%}")
everyone = not_renewed | no_orders | halved
print(f"  all three{len(not_renewed & no_orders & halved):>5,}"
      f" of {len(everyone):>5,}")

# The accounts only B calls churned: what their contracts say.
renewed = ids("""
    SELECT account_id FROM contracts
    WHERE end_date BETWEEN '2024-01-01' AND '2024-06-30'
      AND outcome = 'renewed'""")
no_contract = customers - ids("SELECT account_id FROM contracts")
quiet = no_orders - not_renewed
print(f"\nB flags {len(quiet):,} accounts that A does not")
for name, group in (
    ("renewed a contract in the half-year", quiet & renewed),
    ("no renewal due in the half-year", quiet - renewed - no_contract),
    ("no contract on record at all", quiet & no_contract)):
    print(f"  {name:<37}{len(group):>5,}")
missed = not_renewed - no_orders
print(f"A leavers that B misses (still ordering) {len(missed):>3,}")

# Which definitions flag each account, for the figure.
regions: dict[str, int] = {}
for account in everyone:
    key = "".join(k for k, (_, hit) in defs.items() if account in hit)
    regions[key] = regions.get(key, 0) + 1
Path("code/03/01_label_definitions.json").write_text(json.dumps({
    "customers": len(customers),
    "flagged": {k: len(v[1]) for k, v in defs.items()},
    "regions": dict(sorted(regions.items())),
    "b_not_a": len(quiet), "b_quiet_renewed": len(quiet & renewed),
    "b_quiet_no_contract": len(quiet & no_contract)}, indent=1))
