# The hand-built experiment log at work: three runs recorded, one of
# them a repeat, and the record of one run printed in full.
# timeout: 300
import json
import tempfile
from pathlib import Path

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.tracking import RunLog, code_record, data_record, record
from foresight.tune import (CHAPTER_11, TRAINING, TunableBooster, best,
                            score, search)

table = pd.read_parquet(TABLE)
rows = table[table.end_date.between(*TRAINING)]
tuned = best(search(table))
log = RunLog(Path(tempfile.mkdtemp()) / "runs.jsonl")


def run(name: str, settings: dict) -> dict:
    """Fit on the training period, score on the tuning cohorts, log."""
    model = TunableBooster(**settings).fit(rows)
    s = score(table, settings)
    metrics = {k: s[k] for k in ("log loss", "auc", "leavers")}
    return log.append(record(name, settings, data_record(rows, table),
                             code_record(model), metrics))


run("booster, Chapter 11", CHAPTER_11)
run("booster, tuned", tuned)
run("booster, Chapter 11", CHAPTER_11)          # the same, again

print(f"{'run':<14}{'name':<21}{'data':<10}{'code':<10}{'log loss':>9}")
for r in log.runs():
    print(f"{r['run']:<14}{r['name']:<21}"
          f"{r['data']['content_sha256'][:8]:<10}"
          f"{r['code']['sha256'][:8]:<10}"
          f"{r['metrics']['log loss']:>9.5f}")
ids = [r["run"] for r in log.runs()]
print(f"\nThe repeat has the first run's id: {ids[2] == ids[0]}")
print(f"The log has {len(log.path.read_text().splitlines())} lines;"
      " nothing in it was rewritten.")


def short(v):
    """Hashes cut to 16 characters, floats to 5 figures, to fit."""
    if isinstance(v, dict):
        return {k: short(x) for k, x in v.items()}
    if isinstance(v, list):
        return [short(x) for x in v]
    if isinstance(v, str) and len(v) == 64:
        return v[:16] + "..."
    return round(v, 5) if isinstance(v, float) else v


shown = {k: v for k, v in log.runs()[1].items() if k != "when"}
shown["code"]["environment"] = {k: v for k, v in shown["code"][
    "environment"].items() if k in ("python", "lightgbm", "threads",
                                   "seed")}
print("\n" + json.dumps(short(shown), indent=1))
with open(Path(__file__).with_suffix(".json"), "w") as f:
    json.dump(short(shown), f)
log.path.unlink()
