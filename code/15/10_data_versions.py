# Which table trained which model: the table's hash recomputed from its
# rows and checked against Chapter 4's manifest, a changed copy caught,
# and a run found from the hash alone.
import json
import tempfile
from pathlib import Path

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.tracking import (RunLog, code_record, data_record,
                                record, rows_hash, table_version)
from foresight.tune import CHAPTER_11, TRAINING, TunableBooster

table = pd.read_parquet(TABLE)
v = table_version(table)
print(f"{v['table']}: {len(table):,} rows,"
      f" content {v['content_sha256'][:16]}...")
print(f"  matches its manifest: {v['matches_manifest']}")

# Someone corrects one discount in a copy of the table.
copy = table.copy()
first = copy.discount_pct.first_valid_index()
copy.loc[first, "discount_pct"] += 1
c = table_version(copy)
print(f"\nOne discount changed by one point: content"
      f" {c['content_sha256'][:16]}...")
print(f"  matches the manifest: {c['matches_manifest']}")

# The rows a model is fitted on are a version too.
for name, (a, b) in (("training period", TRAINING),
                     ("training, 2023 only", ("2023-01-01",
                                              "2023-12-31"))):
    r = table[table.end_date.between(a, b)]
    print(f"{name:<22}{len(r):>6,} rows  {rows_hash(r)[:16]}...")

# Two runs, one on each version; find the one the manifest vouches for.
log = RunLog(Path(tempfile.mkdtemp()) / "runs.jsonl")
for name, t in (("on the table", table), ("on the copy", copy)):
    rows = t[t.end_date.between(*TRAINING)]
    model = TunableBooster(**CHAPTER_11).fit(rows)
    log.append(record(name, CHAPTER_11, data_record(rows, t),
                      code_record(model), {}))
manifest = json.loads(TABLE.with_suffix(".manifest.json").read_text())
found = [r["name"] for r in log.runs()
         if r["data"]["content_sha256"] == manifest["content_sha256"]]
print(f"\nRuns fitted on the table the manifest describes: {found}")
log.path.unlink()
