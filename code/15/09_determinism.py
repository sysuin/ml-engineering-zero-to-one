# What the determinism guarantee covers: the order of a sum, the number
# of threads, the seed, and the environment a run records.
import math

import numpy as np
import pandas as pd

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.tracking import environment
from foresight.tune import (CHAPTER_11, TRAINING, TunableBooster, best,
                            score, search)

table = pd.read_parquet(TABLE)
rows = table[table.end_date.between(*TRAINING)]
tuned = best(search(table))


class Seeded(TunableBooster):
    """The booster with its seed and thread count set by hand."""

    def __init__(self, seed=SEED, threads=1, **settings):
        super().__init__(**settings)
        self.seed, self.threads = seed, threads

    def params(self, rounds):
        return {**super().params(rounds), "seed": self.seed,
                "num_threads": self.threads}


# 1. Floating-point addition depends on the order of the terms.
p = Seeded(**tuned).fit(rows).predict_proba(rows)
print("The fitted chances, added up four ways")
for name, total in (("left to right", sum(p.tolist())),
                    ("right to left", sum(p.tolist()[::-1])),
                    ("NumPy, in pairs", float(np.sum(p))),
                    ("exactly, then rounded", math.fsum(p))):
    print(f"  {name:<24}{total!r}")

# 2. Threads: each adds its share of a sum, in whatever order.
q = Seeded(threads=4, **tuned).fit(rows).predict_proba(rows)
print(f"\nOne thread against four: largest difference"
      f" {np.abs(p - q).max():.1e}")

# 3. The seed: the tuned setting samples rows and columns.
print(f"\n{'seed':<10}{'tuned':>9}{'Chapter 11':>12}")
losses = {"tuned": [], "Chapter 11": []}
for k in range(5):
    for name, s in (("tuned", tuned), ("Chapter 11", CHAPTER_11)):
        def make(s=s, seed=SEED + k):
            return Seeded(seed, **s)
        losses[name].append(score(table, make_model=make)["log loss"])
    print(f"SEED + {k:<3}{losses['tuned'][-1]:>9.5f}"
          f"{losses['Chapter 11'][-1]:>12.5f}")
for name, v in losses.items():
    print(f"  {name}: spread {max(v) - min(v):.5f}")

# 4. What a run records about where it ran.
print("\n" + ", ".join(f"{k} {v}" for k, v in environment().items()
                       if k in ("python", "numpy", "lightgbm")))
