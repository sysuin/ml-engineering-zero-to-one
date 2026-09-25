# Exercise 2: a contract for account_history, the table Chapter 4 reads
# for each account's segment on the morning of its mark, checked
# against the warehouse.
from contextlib import closing

from foresight import contracts
from foresight.config import ML_WAREHOUSE
from foresight.contracts import Column, Contract, Rule
from foresight.contracts.orders import DAY
from foresight.data.expectations import SEGMENTS

HISTORY = Contract(
    table="account_history",
    owner="CRM team",
    version="0.1",
    columns=(
        Column("account_id", "integer", refers="accounts.account_id"),
        Column("valid_from", "text", form=DAY),
        Column("valid_to", "text", required=False, form=DAY),
        Column("segment", "text", values=tuple(sorted(SEGMENTS))),
        Column("account_manager", "text"),
    ),
    key=("account_id", "valid_from"),
    rules=(
        Rule("a period ends on or after the day it starts",
             "SELECT COUNT(*) FROM {rows} WHERE valid_to < valid_from"),
        Rule("each account has exactly one open period",
             "SELECT COUNT(*) FROM (SELECT account_id FROM {rows}"
             " GROUP BY account_id"
             " HAVING SUM(valid_to IS NULL) <> 1)"),
        Rule("no period starts after the record ends",
             "SELECT COUNT(*) FROM {rows}"
             " WHERE valid_from > '2025-12-31'"),
    ),
    words=("a change is a new row; an old row is closed, never edited",),
)

print(contracts.describe(HISTORY) + "\n")
with closing(contracts.read_only(ML_WAREHOUSE)) as con:
    found = contracts.breaches(con, HISTORY)
    for b in found:
        print(b.line())
    print(f"{contracts.clauses(HISTORY)} clauses, {len(found)} broken")
