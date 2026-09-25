# Three o'clock in the morning: a run that fails half way, the retry,
# the run after that, a second run while one holds the lock, a lock
# left by a run that died, and the database refusing a second list.
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from foresight.impact import holdout
from foresight.serve import SANDBOX, batch, store

root = Path(tempfile.mkdtemp())
shutil.copytree(SANDBOX / "artifacts", root / "artifacts")
db, lock = store.path(root), store.path(root).parent / batch.LOCK
record = holdout.record


def full_disk(*args, **kwargs):
    raise OSError(28, "No space left on device")


holdout.record = full_disk                 # the disk fills at step 5
try:
    batch.month("2025-12-31", root=root)
except OSError as e:
    print(f"run failed: {e}")
holdout.record = record                    # somebody clears the disk
print(f"  lock left behind: {lock.exists()}; scores written:"
      f" {store.read(db, 'SELECT COUNT(*) AS n FROM scores').n[0]}")

done = batch.month("2025-12-31", root=root)
print(f"retry: run {done['run']} {done['status']},"
      f" {len(done['scored'])} rows")
print(batch.summary(batch.month("2026-01-01", root=root)))

lock.write_text(f"{socket.gethostname()} {os.getpid()}")  # held
try:
    batch.month("2026-01-01", root=root)
except batch.Locked as e:
    print(f"while another run works: {e}")
gone = subprocess.Popen([sys.executable, "-c", "pass"])
gone.wait()
lock.write_text(f"{socket.gethostname()} {gone.pid}")     # died
after = batch.month("2026-01-01", root=root)
print(f"a dead run's lock: taken over, {after['status']};"
      f" lock left: {lock.exists()}")

with closing(sqlite3.connect(db)) as con:
    try:
        with con:
            con.execute("INSERT INTO runs (job, mark, run_on, status)"
                        " VALUES ('renewal', '2025-12-31',"
                        " '2026-01-02', 'finished')")
    except sqlite3.IntegrityError as e:
        print(f"a second list, written by hand:\n  {e}")

print("\nThe runs table")
for r in store.read(db, "SELECT * FROM runs").itertuples():
    print(f"  run {r.run}  {r.mark}  {r.status:<9}step {r.step}")
    if isinstance(r.error, str):
        print(f"        {r.error}")
shutil.rmtree(root)
