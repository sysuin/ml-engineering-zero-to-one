# Exercise 1: two ways the warehouse can change under the job, and
# what the job does about each. Both runs fail, and say where.
import shutil
import tempfile
import textwrap
from pathlib import Path

from foresight.pipeline import features
from foresight.serve import SANDBOX, batch, online, store


def say(what, e):
    """The error's type, then its message wrapped to the page."""
    print(f"{what}: {type(e).__name__}:")
    print(textwrap.fill(str(e), 66, initial_indent="  ",
                        subsequent_indent="  "))


root = Path(tempfile.mkdtemp())
shutil.copytree(SANDBOX / "artifacts", root / "artifacts")
at_mark, rows = features.at_mark, online.rows

# 1. A load starts writing discounts as decimals: 5.0, not 5.
features.at_mark = lambda *a: at_mark(*a).astype({"discount_pct":
                                                   float})
try:
    batch.month("2025-12-31", root=root)
except Exception as e:
    say("decimal discounts", e)


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
    say("a new segment", e)
features.at_mark, online.rows = at_mark, rows

print()
db = store.path(root)
for r in store.read(db, "SELECT * FROM runs").itertuples():
    print(f"run {r.run}: {r.status} at step {r.step}")
n = store.read(db, "SELECT COUNT(*) AS n FROM scores").n[0]
print(f"scores written: {n}")
shutil.rmtree(root)
