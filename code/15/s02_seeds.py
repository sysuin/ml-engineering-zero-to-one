# Exercise 2: the tuned setting and Chapter 11's under ten seeds each.
# timeout: 300
import numpy as np
import pandas as pd

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.tune import (CHAPTER_11, TunableBooster, best, score,
                            search)

table = pd.read_parquet(TABLE)
tuned = best(search(table))


class Seeded(TunableBooster):
    def __init__(self, seed, **settings):
        super().__init__(**settings)
        self.seed = seed

    def params(self, rounds):
        return {**super().params(rounds), "seed": self.seed}


losses = {}
for name, s in (("tuned", tuned), ("Chapter 11", CHAPTER_11)):
    losses[name] = np.array([
        score(table, make_model=lambda s=s, k=k: Seeded(SEED + k, **s))
        ["log loss"] for k in range(10)])
    v = losses[name]
    print(f"{name:<12} mean {v.mean():.5f}  sd {v.std(ddof=1):.5f}"
          f"  range {v.min():.5f} to {v.max():.5f}")
gap = losses["Chapter 11"].mean() - losses["tuned"].mean()
print(f"\nMean gain of the tuned setting: {gap:.5f}")
print(f"Seeds where the tuned setting beat Chapter 11's:"
      f" {(losses['tuned'] < losses['Chapter 11']).sum()} of 10")
