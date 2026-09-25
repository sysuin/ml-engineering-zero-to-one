# expect-fail
# The job a month on, on 30 January 2026, when the warehouse's load
# has stopped at the end of December: it refuses, before any scoring.
import shutil
import tempfile
from pathlib import Path

from foresight.serve import SANDBOX, batch

with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(SANDBOX / "artifacts", Path(tmp) / "artifacts")
    batch.main(["--on", "2026-01-30", "--root", tmp])
