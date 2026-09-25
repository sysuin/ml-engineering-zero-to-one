# Grid search against random search on two of the booster's settings,
# the smallest leaf and the learning rate: nine trials each, scored on
# the tuning cohorts. Then ten more random searches of nine.
# timeout: 300
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.tune import draw, grid, score

table = pd.read_parquet(TABLE)
TWO = {"min_leaf": (5, 400, True, True),
       "rate": (0.02, 0.3, True, False)}


def run(settings):
    """Score each setting; the rest stay at Chapter 11's."""
    return [{**s, "log loss": score(table, s)["log loss"]}
            for s in settings]


# Three values of each, evenly spaced on the log scale, end to end.
by_grid = run(grid({"min_leaf": [5, 45, 400],
                    "rate": [0.02, 0.077, 0.3]}))
by_chance = run(draw(9, SEED, TWO))

print(f"{'grid':>24}{'':8}{'random':>22}")
print(f"{'min leaf':>10}{'rate':>7}{'log loss':>10}{'':8}"
      f"{'min leaf':>10}{'rate':>7}{'log loss':>10}")
for g, r in zip(by_grid, by_chance):
    print(f"{g['min_leaf']:>10}{g['rate']:>7.3f}{g['log loss']:>10.5f}"
          f"{'':8}{r['min_leaf']:>10}{r['rate']:>7.3f}"
          f"{r['log loss']:>10.5f}")
best_grid = min(t["log loss"] for t in by_grid)
best_chance = min(t["log loss"] for t in by_chance)
print(f"\n{'best':>17}{best_grid:>10.5f}{'':25}{best_chance:>10.5f}")
for name, trials in (("grid", by_grid), ("random", by_chance)):
    print(f"{name + ':':<8}{len({t['min_leaf'] for t in trials})}"
          f" values of min leaf,"
          f" {len({t['rate'] for t in trials})} of the rate")

# One random search is one draw. Ten more, each of nine trials.
more = [run(draw(9, SEED + k, TWO)) for k in range(1, 11)]
bests = np.array([min(t["log loss"] for t in m) for m in more])
print(f"\nTen more random searches of nine: best log loss"
      f" {bests.min():.5f} to {bests.max():.5f}")
print(f"  better than the grid's best in {(bests < best_grid).sum()}"
      f" of 10, median {np.median(bests):.5f}")

with open(Path(__file__).with_suffix(".json"), "w") as f:
    json.dump({"grid": by_grid, "random": by_chance,
               "more": [t for m in more for t in m]}, f)
