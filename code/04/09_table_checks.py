# The builder's checks, run on the real table and on two broken ones.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import (
    FORESIGHT_DATA, TableCheckError, build, check, expected_rows, write)

FIRST, LAST = "2023-01-01", "2024-06-30"
table = build(FIRST, LAST)
expected = expected_rows(FIRST, LAST)
print(f"The table as built: {len(table):,} rows")
print(f"  problems: {check(table, FIRST, LAST, None, expected)}")

# Broken 1: segment joined from account_history on the account alone.
con = sqlite3.connect(ML_WAREHOUSE)
history = pd.read_sql_query(
    "SELECT account_id, segment AS segment_h FROM account_history", con)
fanned = table.merge(history, on="account_id")
print(f"\nSegment joined without dates: {len(fanned):,} rows")
for problem in check(fanned, FIRST, LAST, None, expected):
    print(f"  {problem}")

# Broken 2: retraining on 1 January 2024, rows chosen by their moment.
DAY = "2024-01-01"
early = table[table.moment < DAY]
known = expected_rows(FIRST, LAST, DAY)
print(f"\nRows whose moment passed before {DAY}: {len(early):,}")
for problem in check(early, FIRST, LAST, DAY, known):
    print(f"  {problem}")

# write() runs the same checks, and a failure leaves no file behind.
path = FORESIGHT_DATA / "broken_table.parquet"
try:
    write(fanned, FIRST, LAST, path=path)
except TableCheckError as refusal:
    problems = str(refusal).split("; ")
    print(f"\nwrite() refused: {len(problems)} problems")
print(f"  {path.name} exists: {path.exists()}")
