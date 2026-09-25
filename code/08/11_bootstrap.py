# How sure is 23.8%? Three ways to resample the validation cohorts.
import pandas as pd

from foresight.config import rng
from foresight.data.build_table import TABLE
from foresight.evaluate import (REPS, SPLITS, backtest, bootstrap,
                                interval, measure)

table = pd.read_parquet(TABLE)
for split, (first, last) in SPLITS.items():
    ids = table[table.end_date.between(first, last)].account_id
    print(f"{split}: {len(ids):,} contracts from {ids.nunique():,}"
          " accounts")

scored = backtest(table, *SPLITS["validation"])
point = measure(scored)["precision"]["model"]
print(f"\nThe model's precision at 40 a cohort: {point:.1%}")


def ignoring_cohorts(g):
    """Draw 2,083 contracts from all of them: cohort sizes wander."""
    return scored.iloc[g.integers(0, len(scored), len(scored))]


def whole_cohorts(g):
    """Draw six whole cohorts from the six, with replacement."""
    marks = scored.moment.unique()
    parts = [scored[scored.moment == m].assign(moment=i)
             for i, m in enumerate(g.choice(marks, len(marks)))]
    return pd.concat(parts)


ways = {"contracts within cohorts":
        bootstrap(scored)[("precision", "model")]}
for name, draw in (("contracts, any cohort", ignoring_cohorts),
                   ("whole cohorts", whole_cohorts)):
    g = rng()
    ways[name] = [measure(draw(g))["precision"]["model"]
                  for _ in range(REPS)]
print(f"{REPS:,} resamples each{'95% interval':>21}{'width':>9}")
for name, v in ways.items():
    lo, hi = interval(v)
    print(f"  {name:<26}{lo:>7.1%} to {hi:.1%}{hi - lo:>9.1%}")
