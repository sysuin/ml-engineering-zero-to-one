# Entity resolution: accounts the CRM migration split across two ids.
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import legacy_ids, tidy

con = sqlite3.connect(ML_WAREHOUSE)
q = con.execute

print("accounts by source   rows   lowest id   highest id")
for src, n, lo, hi in q("""SELECT crm_source, COUNT(*), MIN(account_id),
                           MAX(account_id) FROM accounts GROUP BY 1"""):
    print(f"  {src:15}{n:>8,}{lo:>12}{hi:>13}")
n, last = q("""SELECT COUNT(*), MAX(order_date) FROM orders
               WHERE account_id >= 90000""").fetchone()
print(f"Orders under a legacy id: {n:,}, the last on {last}")

# Candidate pairs: a legacy row and a current row, perhaps one firm.
pairs = pd.read_sql_query("""
    SELECT old.account_id AS legacy_id, new.account_id,
           old.name AS legacy_name, new.name,
           old.since = new.since AS same_since
    FROM accounts old JOIN accounts new
      ON new.postcode = old.postcode AND new.crm_source = 'Meridian CRM'
    WHERE old.crm_source = 'Legacy CRM'""", con)
since = pairs[pairs.same_since == 1]
print("\nCandidate pairs")
print(f"  same postcode                          {len(pairs):>4}")
print(f"  ...and the same start date             {len(since):>4}")
exact = (since.legacy_name == since.name).sum()
print(f"  ...and exactly the same name           {exact:>4}")
tidied = (since.legacy_name.map(tidy) == since.name.map(tidy)).sum()
print(f"  ...and the same name once tidied       {tidied:>4}")
kinds = {"upper case": since.legacy_name.str.isupper(),
         "' Inc' added": since.legacy_name.str.endswith(" Inc"),
         "a double space": since.legacy_name.str.contains("  "),
         "' (old)' added": since.legacy_name.str.endswith(" (old)")}
for kind, hit in kinds.items():
    print(f"    {kind:38}{hit.sum():>4}")

ids = legacy_ids(con)
print(f"legacy_ids(): {len(ids)} legacy ids -> "
      f"{len(set(ids.values()))} current ids")

# The damage: a year of orders before each moment, counted both ways.
orders = pd.read_sql_query("SELECT account_id, order_date FROM orders",
                           con, parse_dates=["order_date"])
orders["resolved_id"] = orders.account_id.replace(ids)
renewals = pd.read_sql_query("""
    SELECT contract_id, account_id,
           date(end_date, '-90 days') AS moment,
           outcome = 'not_renewed' AS left_
    FROM contracts
    WHERE end_date BETWEEN '2023-01-01' AND '2024-06-30'
      AND outcome IS NOT NULL""", con, parse_dates=["moment"])
renewals = renewals[renewals.account_id.isin(ids.values())]


def year_of_orders(key: str) -> pd.Series:
    """Orders in the year before each moment, matching on `key`."""
    x = renewals.merge(orders, left_on="account_id", right_on=key)
    age = (x.moment - x.order_date).dt.days
    x = x[(age >= 1) & (age <= 365)]
    return (x.groupby("contract_id").size()
             .reindex(renewals.contract_id, fill_value=0))


raw, fixed = year_of_orders("account_id"), year_of_orders("resolved_id")
late = renewals.set_index("contract_id").moment >= "2024-03-01"
print(f"\nTraining renewals of the split accounts: {len(renewals)}"
      f" ({renewals.left_.sum()} not renewed)")
print(f"  {'orders in the year before the moment':38}{'raw':>6}"
      f"{'resolved':>10}")
none = (raw == 0).sum(), (fixed == 0).sum()
print(f"  {'  none at all':38}{none[0]:>6}{none[1]:>10}")
print(f"  {'  mean, moments before the migration':38}"
      f"{raw[~late].mean():>6.1f}{fixed[~late].mean():>10.1f}")
print(f"  {f'  mean, the {late.sum()} moments after it':38}"
      f"{raw[late].mean():>6.1f}{fixed[late].mean():>10.1f}")

# Chapter 3's 180 'churned' customers with no contract on record.
quiet = pd.read_sql_query("""
    SELECT account_id FROM accounts
    WHERE account_id NOT IN (SELECT account_id FROM contracts)
      AND account_id IN (SELECT account_id FROM orders WHERE order_date
                         BETWEEN '2023-01-01' AND '2023-12-31')""", con)
later = orders[(orders.order_date >= "2024-04-01")
               & (orders.order_date <= "2024-06-30")]
still = quiet.account_id.map(ids).isin(later.account_id)
print(f"\nChapter 3's customers with no contract: {len(quiet)}")
print(f"  of them legacy ids                     "
      f"{quiet.account_id.isin(ids).sum():>4}")
print(f"  ordering in Q2 2024 under the new id   {still.sum():>4}")

# One company, two ids: the figure's example.
old, new = 90058, ids[90058]
seen = orders[orders.resolved_id == new]
seen = seen[(seen.order_date >= "2023-04-01")
            & (seen.order_date < "2024-04-01")]
seen = seen.sort_values("order_date")
Path("code/04/06_resolve_entities.json").write_text(json.dumps({
    "legacy_id": old, "account_id": new,
    "rows": {i: rest for i, *rest in q(
        "SELECT account_id, name, postcode, since, crm_source "
        "FROM accounts WHERE account_id IN (?, ?)", (old, new))},
    "orders": [[int(a), str(d.date())] for a, d in
               zip(seen.account_id, seen.order_date)]}, indent=1))
