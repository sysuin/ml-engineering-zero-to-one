# Which region goes first. The rule is written before the table: the
# pilot needs the most calls on a list, so that faults show up in the
# first month, and a leaving rate within a point of the company's, so
# that what it teaches carries to the other four regions.
import shutil
import sqlite3
from contextlib import closing

import pandas as pd

from foresight import demo
from foresight.config import ML_WAREHOUSE
from foresight.serve import batch

with closing(sqlite3.connect(ML_WAREHOUSE)) as con:
    year = pd.read_sql_query("""
        SELECT r.name AS region, c.outcome,
               date(c.end_date, '-90 days') AS mark
        FROM contracts c JOIN accounts a USING (account_id)
        JOIN regions r USING (region_id)
        WHERE a.is_key_account = 0
          AND date(c.end_date, '-90 days')
              BETWEEN '2025-01-01' AND '2025-12-31'""", con)

known = year.dropna(subset=["outcome"])
company = (known.outcome == "not_renewed").mean()
lists = year.mark.nunique()
by = year.groupby("region").agg(contracts=("mark", "size"))
by["a_list"] = by.contracts / lists
by["left"] = (known.assign(left=known.outcome == "not_renewed")
              .groupby("region").left.mean())

root = demo.workspace()                  # the list, made afresh
scored = batch.month("2025-12-31", root=root)["scored"]
shutil.rmtree(root)
latest = scored[scored.listed]          # region is one of its columns
by["calls"] = latest.groupby("region").size()
by["held"] = latest[latest.arm == "held out"].groupby("region").size()

print(f"Long-tail contracts at a 2025 mark, {lists} lists; the"
      f" latest\nlist's 40 calls (31 December) by region\n")
print(f"{'region':<11}{'a list':>8}{'left':>8}{'calls':>7}"
      f"{'held out':>10}")
for region, r in by.iterrows():
    print(f"{region:<11}{r.a_list:>8.0f}{r.left:>8.2%}"
          f"{r.calls:>7.0f}{r.held:>10.0f}")
print(f"{'company':<11}{by.a_list.sum():>8.0f}{company:>8.2%}"
      f"{by.calls.sum():>7.0f}{by.held.sum():>10.0f}")

typical = by[(by.left - company).abs() <= 0.01]
pilot = typical.calls.idxmax()
print(f"\nWithin a point of the company's rate:"
      f" {', '.join(typical.index)}")
print(f"The most calls among them: {pilot},"
      f" {by.calls[pilot]:.0f} of 40")
