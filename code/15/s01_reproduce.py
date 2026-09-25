# Exercise 1: rebuild the tuned booster's run from its record alone,
# and check that the rebuilt run has the recorded id.
import json
from pathlib import Path

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.tracking import (code_record, data_record, record,
                                rows_hash)
from foresight.tune import TunableBooster, score

saved = json.loads(Path(__file__).with_name("07_tiny_tracker.json")
                   .read_text())
table = pd.read_parquet(TABLE)
same = rows_hash(table)[:16] == saved["data"]["content_sha256"][:16]
print(f"The table on disk is the one recorded: {same}")

# The record says how many rows, and their hash, but not which rows.
# Try the windows a Foresight model could have been fitted on.
windows = {"training period": ("2023-01-01", "2024-06-30"),
           "2023 only": ("2023-01-01", "2023-12-31"),
           "everything to 2024": ("2023-01-01", "2024-12-31")}
for name, (a, b) in windows.items():
    rows = table[table.end_date.between(a, b)]
    hit = rows_hash(rows)[:16] == saved["data"]["rows_sha256"][:16]
    print(f"  {name:<20}{len(rows):>6,} rows  matches: {hit}")
    if hit:
        fitted = rows

full = saved["params"]            # as printed: rounded to 5 figures
print(f"\nRecorded rate, as printed: {full['rate']}")
model = TunableBooster(**full).fit(fitted)
s = score(table, full)
metrics = {k: s[k] for k in ("log loss", "auc", "leavers")}
print(f"Rebuilt log loss {metrics['log loss']:.5f}, recorded"
      f" {saved['metrics']['log loss']}")
again = record(saved["name"], full, data_record(fitted, table),
               code_record(model), metrics)
print(f"Rebuilt id {again['run']}, recorded {saved['run']}:"
      f" {again['run'] == saved['run']}")
