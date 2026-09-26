# Exercise 3: the checklist's worst gap closed. On a morning whose
# mark is due, no finished list for it is an alert; tested on a copy
# of the sandbox where the night's run failed, then succeeded.
import shutil

import pandas as pd

from foresight import demo
from foresight.monitor.alerts import Alert
from foresight.pipeline import features
from foresight.serve import batch, store


def late_list(root, day) -> Alert | None:
    """An alert if the mark due on `day` has no finished list."""
    mark = features.mark_on(day)
    done = pd.DataFrame()
    if store.path(root).exists():
        done = store.read(store.path(root), "SELECT run FROM runs WHERE"
                          " job = 'renewal' AND mark = ? AND status ="
                          " 'finished'", (f"{mark:%Y-%m-%d}",))
    if len(done):
        return None
    return Alert(pd.Timestamp(day), "no list", f"{mark:%Y-%m-%d}",
                 (pd.Timestamp(day) - mark).days, 0,
                 "due, and no finished list")


root = demo.workspace()
morning = "2025-12-31"
try:                                    # a holdout that cannot be drawn
    batch.month(morning, root=root, held=-1)
except ValueError as e:
    print(f"night of the mark: the job failed ({type(e).__name__})")


def check(when, day):
    a = late_list(root, day)
    said = (f"ALERT, {a.subject} {a.note}" if a
            else "quiet, the list is made")
    print(f"{when:<22}{said}")


check("8 a.m., the mark", morning)
batch.month(morning, root=root)         # someone reruns it
print("9 a.m.: rerun by hand, finished")
check("8 a.m., the next day", "2026-01-01")
shutil.rmtree(root)
