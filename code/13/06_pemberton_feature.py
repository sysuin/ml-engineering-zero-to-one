# The Pemberton feature, built point in time from ticket text: Quality
# or Delivery tickets naming the cloth, MRD-CLE-001, before the mark.
# No outcome is read here; §13.11 reads the test year once.
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE, legacy_ids
from foresight.evaluate import SPLITS
from foresight.features.sources import Context, Sources

con = sqlite3.connect(ML_WAREHOUSE)
tickets = pd.read_sql_query("""SELECT account_id, date(opened_at) AS
    day, category, body FROM tickets""", con)
tickets["account_id"] = tickets.account_id.replace(legacy_ids(con))
tickets["day"] = pd.to_datetime(tickets.day)
cloth = tickets[tickets.category.isin(["Quality", "Delivery"])
                & tickets.body.str.contains("MRD-CLE-001", case=False)]

month = cloth.day.dt.to_period("M")
before = month < pd.Period("2024-10")
print(f"Cloth complaints a month: {before.sum() / 33:.1f} on average,"
      " January 2022 to\nSeptember 2024; then")
for m, n in month[~before].value_counts().sort_index().items():
    if m <= pd.Period("2025-03"):
        print(f"  {m}  {n:>4}")


def pemberton(rows, days):
    """Cloth complaints in the `days` before each contract's mark."""
    none = pd.DataFrame(columns=["account_id", "day"])
    ctx = Context(rows, Sources(none, none, cloth, none))
    e = ctx.window("tickets", days)
    return ctx.per_contract(e.groupby("contract_id").size()).to_numpy()


table = pd.read_parquet(TABLE)
table["pemberton_90d"] = pemberton(table, 90)
table["pemberton_365d"] = pemberton(table, 365)
table["split"] = "train"
for name, (first, last) in SPLITS.items():
    table.loc[table.end_date.between(first, last), "split"] = name
q = table.assign(quarter=table.moment.dt.to_period("Q"))
print("\nContracts with a complaint before the mark")
print(f"{'marks in':<10}{'split':<12}{'contracts':>9}{'90 days':>9}"
      f"{'365 days':>10}")
out = []
for (quarter, split), c in q.groupby(["quarter", "split"]):
    n90 = (c.pemberton_90d > 0).sum()
    n365 = (c.pemberton_365d > 0).sum()
    print(f"{str(quarter):<10}{split:<12}{len(c):>9,}{n90:>9}"
          f"{n365:>10}")
    out.append([str(quarter), split, len(c), int(n90), int(n365)])
seen = table[table.split != "test"]
print(f"\nTraining and validation: {(seen.pemberton_90d > 0).sum()} of"
      f" {len(seen):,} contracts carry the\n90-day column")
with open("code/13/06_pemberton_feature.json", "w") as f:
    json.dump({"quarters": out}, f)
