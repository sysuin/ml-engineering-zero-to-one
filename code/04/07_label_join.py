# Three ways the label join lets the future in, counted on Meridian.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
TRAIN = "c.end_date BETWEEN '2023-01-01' AND '2024-06-30'"

right = pd.read_sql_query(f"""
    SELECT c.contract_id, c.outcome = 'not_renewed' AS label
    FROM contracts c WHERE {TRAIN} AND c.outcome IS NOT NULL""", con)
print(f"Training renewals, one label each: {len(right):,}"
      f"  (label rate {right.label.mean():.1%})")

# 1. The label joined on the account: every outcome it ever had.
fanned = pd.read_sql_query(f"""
    SELECT c.contract_id, l.end_date AS label_end,
           l.outcome = 'not_renewed' AS label
    FROM contracts c
    JOIN contracts l ON l.account_id = c.account_id
    WHERE {TRAIN} AND c.outcome IS NOT NULL
      AND l.outcome IS NOT NULL""", con)
print("\n1. Join the label on the account instead of the contract")
print(f"   rows after the join                    {len(fanned):>6,}")
copies = fanned.contract_id.value_counts()
many = (copies > 1).sum()
print(f"   contracts on more than one row         {many:>6,}")

# 2. The usual repair: one row per contract, keeping the worst outcome.
worst = fanned.groupby("contract_id").label.max()
both = right.join(worst.rename("worst"), on="contract_id")
flipped = both[(both.label == 0) & (both.worst == 1)]
gone = fanned[fanned.label == 1].groupby("contract_id").label_end.max()
after = (gone.reindex(flipped.contract_id) > "2024-06-30").sum()
print("2. ...then keep each contract's worst outcome")
print(f"   renewals relabelled 'not renewed'      {len(flipped):>6,}")
print(f"     by a departure after June 2024       {after:>6,}")
print(f"   label rate                   {right.label.mean():>6.1%}"
      f" -> {both.worst.mean():.1%}")

# 3. Retraining on 1 January 2024, choosing rows by their moment.
DAY = "2024-01-01"
by_moment, by_end = con.execute(f"""
    SELECT SUM(date(c.end_date, '-90 days') < :day),
           SUM(c.end_date < :day)
    FROM contracts c
    WHERE {TRAIN} AND c.outcome IS NOT NULL""", {"day": DAY}).fetchone()
print(f"3. Retrain on {DAY}, keeping contracts whose moment has passed")
print(f"   rows kept                              {by_moment:>6,}")
print(f"   outcome not yet recorded on {DAY}  {by_moment - by_end:>6,}")
