# The three feeds behind the table, against their contracts, on the
# last morning of the record; then the same feeds on a bad night, made
# by shadowing two tables with temporary views that change a few rows.
from contextlib import closing

from foresight import contracts
from foresight.config import ML_WAREHOUSE

ON = "2025-12-31"
print(f"the feeds on the morning of {ON}:")
with closing(contracts.read_only(ML_WAREHOUSE)) as con:
    for c in contracts.FEEDS:
        found = contracts.breaches(con, c, ON)
        print(f"  {c.table:<12} {c.owner:<20}"
              f"{contracts.clauses(c):>3} clauses, {len(found)} broken")

BAD_NIGHT = """
CREATE TEMP VIEW orders AS                  -- a new loader, and half
SELECT order_id, account_id,                -- of the 30th missing
       CASE WHEN order_date = '2025-12-30'
            THEN '12/30/2025' ELSE order_date END AS order_date,
       channel
FROM main.orders
WHERE order_date <> '2025-12-30' OR order_id % 2 = 0;
CREATE TEMP VIEW tickets AS                 -- the desks' new channel
SELECT ticket_id, account_id, opened_at,
       CASE WHEN opened_at >= '2025-12-29' THEN 'Chat'
            ELSE channel END AS channel,
       sku, language, body, desk, category, priority
FROM main.tickets;
"""
print("\nthe same feeds on a bad night:")
with closing(contracts.read_only(ML_WAREHOUSE)) as con:
    con.executescript(BAD_NIGHT)
    for c in contracts.FEEDS:
        for b in contracts.breaches(con, c, ON):
            print(f"  {b.table}: {b.clause}")
            print(f"    {b.evidence()}")
