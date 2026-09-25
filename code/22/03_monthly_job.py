# The monthly job on the morning of 31 December 2025, as `make score`
# runs it, and again the next morning, when it has nothing to do.
import json

import pandas as pd

from foresight.pipeline import features
from foresight.serve import SANDBOX, batch, store

folder = store.path(SANDBOX).parent
for name in (store.FILE, "holdout.csv", batch.LOCK):  # start clean
    (folder / name).unlink(missing_ok=True)

first = batch.month("2025-12-31", root=SANDBOX)
print(batch.summary(first, ranks=[1, 2, 40]))

print()
print(batch.summary(batch.month("2026-01-01", root=SANDBOX)))

runs = store.read(store.path(SANDBOX), """
    SELECT run, mark, run_on, source, model_as_of, status, rows
    FROM runs""")
print("\n" + runs.to_string(index=False))

mark, scored = first["mark"], first["scored"]
end = mark + pd.Timedelta(days=90)
after = features.mark_on(mark + pd.Timedelta(days=45))
arms = scored.arm.value_counts()
with open("code/22/03_monthly_job.json", "w") as f:
    json.dump({"mark": f"{mark:%Y-%m-%d}", "contracts": len(scored),
               "listed": int(scored.listed.sum()),
               "called": int(arms["called"]),
               "held": int(arms["held out"]),
               "end": f"{end:%Y-%m-%d}", "end_day": 90,
               "notice_day": 30, "next_mark": f"{after:%Y-%m-%d}",
               "next_day": (after - mark).days}, f)
