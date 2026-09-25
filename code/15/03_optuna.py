# Optuna's TPE sampler against random search: forty trials each over
# six of the booster's settings on the tuning cohorts, three seeds each.
# timeout: 600
import json
from pathlib import Path

import pandas as pd

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.tune import CHAPTER_11, Budget, score, search

table = pd.read_parquet(TABLE)
forty = Budget(trials=40, patience=40)      # no early stop: 40 each
runs = {(sampler, k): search(table, forty, sampler, SEED + k)
        for sampler in ("tpe", "random") for k in range(3)}
start = score(table)["log loss"]

tpe, rnd = runs["tpe", 0], runs["random", 0]
print(f"Chapter 11's setting: {start:.5f}\n")
print(f"{'best log loss after':<22}{'TPE':>9}{'random':>9}")
for n in (5, 10, 15, 20, 30, 40):
    print(f"  {n:>2} trials{'':<11}{tpe['best so far'][n - 1]:>9.5f}"
          f"{rnd['best so far'][n - 1]:>9.5f}")
later = slice(10, 40)                       # after TPE's first ten
print(f"{'median, trials 11-40':<22}"
      f"{tpe['log loss'][later].median():>9.5f}"
      f"{rnd['log loss'][later].median():>9.5f}")
print(f"{'two leaves, 11-40':<22}"
      f"{(tpe.leaves[later] == 2).sum():>9}"
      f"{(rnd.leaves[later] == 2).sum():>9}")

best = tpe.loc[tpe["log loss"].idxmin()]
print(f"\nTPE's best, trial {int(best.trial)}:")
for part in (list(CHAPTER_11)[:3], list(CHAPTER_11)[3:]):
    print("  " + ", ".join(f"{k} {best[k]:.3g}" for k in part))

print(f"\n{'seed':<10}{'TPE best':>10}{'random best':>13}")
for k in range(3):
    print(f"SEED + {k:<3}{runs['tpe', k]['log loss'].min():>10.5f}"
          f"{runs['random', k]['log loss'].min():>13.5f}")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    kept = ["trial", "log loss", "best so far", "leaves"]
    json.dump({"start": start,
               "runs": {f"{s} {k}": r[kept].to_dict("list")
                        for (s, k), r in runs.items()}}, f)
