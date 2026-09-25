# Exercise 4: a second list for a month already made, with a reason,
# and what it does to the record, the answers and the holdout.
import shutil
import tempfile
from pathlib import Path

from foresight.serve import SANDBOX, batch, store

root = Path(tempfile.mkdtemp())
shutil.copytree(SANDBOX / "artifacts", root / "artifacts")
shutil.copytree(store.path(SANDBOX).parent, store.path(root).parent)
db = store.path(root)
before = store.read(db, "SELECT contract_id, arm FROM answers"
                    " WHERE listed = 1").set_index("contract_id")

again = batch.month("2025-12-31", root=root,
                    again="finance corrected two discounts")
print(f"run {again['run']}: {again['status']}")
runs = store.read(db, "SELECT run, status, reason FROM runs")
for r in runs.fillna("").itertuples():
    print(f"  run {r.run}  {r.status:<9}{r.reason}".rstrip())

after = store.read(db, "SELECT run, contract_id, arm FROM answers"
                   " WHERE listed = 1").set_index("contract_id")
kept = store.read(db, "SELECT COUNT(*) AS n FROM scores WHERE run = 1")
print(f"\nanswers now come from run {after.run.iloc[0]};"
      f" run 1's {kept.n[0]} rows are still there")
same = (before.arm == after.arm.reindex(before.index)).sum()
print(f"holdout arms unchanged for {same} of {len(before)} listed")
lines = (store.path(root).parent / "holdout.csv").read_text()
print(f"holdout file: {len(lines.splitlines()) - 1} rows, as drawn")
shutil.rmtree(root)
