# Exercise 1: two ways the warehouse can change under the job, and
# what the job does about each. Both runs fail, and say where.
import shutil
import tempfile
from pathlib import Path

from foresight.pipeline import features
from foresight.serve import SANDBOX, batch, online, store

root = Path(tempfile.mkdtemp())
shutil.copytree(SANDBOX / "artifacts", root / "artifacts")
at_mark, rows = features.at_mark, online.rows

# 1. A load starts writing discounts as decimals: 5.0, not 5.
features.at_mark = lambda *a: at_mark(*a).astype({"discount_pct":
                                                   float})
try:
    batch.month("2025-12-31", root=root)
except Exception as e:
    print(f"decimal discounts: {type(e).__name__}:\n  {e}")


# 2. The CRM starts calling some small businesses Wholesale.
def wholesale(frame):
    frame = frame.copy()
    small = frame.segment == "Small business"
    frame.loc[small & (frame.contract_id % 2 == 0), "segment"] = \
        "Wholesale"
    return frame


features.at_mark = lambda *a: wholesale(at_mark(*a))
online.rows = lambda *a: wholesale(rows(*a))
try:
    batch.month("2025-12-31", root=root)
except Exception as e:
    print(f"a new segment: {type(e).__name__}:\n  {e}")
features.at_mark, online.rows = at_mark, rows

print()
db = store.path(root)
for r in store.read(db, "SELECT * FROM runs").itertuples():
    print(f"run {r.run}: {r.status} at step {r.step}")
n = store.read(db, "SELECT COUNT(*) AS n FROM scores").n[0]
print(f"scores written: {n}")
shutil.rmtree(root)
