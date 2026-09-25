# Foresight's tuning budget: sixty trials at most, stopped sooner by
# twenty in a row that improve on the best by no more than Chapter
# 11's line of noise. Then the same search left to run all sixty.
# timeout: 300
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.tune import BUDGET, NOISE, Budget, best, search

table = pd.read_parquet(TABLE)
print(f"Budget: {BUDGET.trials} trials, patience {BUDGET.patience},"
      f" {BUDGET.minutes:g} minutes; noise {NOISE}\n")
trials = search(table)

print(f"{'trial':>5}{'log loss':>10}  a new best by more than {NOISE}")
level = float("inf")
for n, loss in zip(trials.trial, trials["log loss"]):
    if loss < level - NOISE:
        level = loss
        print(f"{n:>5}{loss:>10.5f}")
top = trials.loc[trials["log loss"].idxmin()]
print(f"\nStopped after trial {len(trials)}, by"
      f" {trials.attrs['stopped by']}; best {top['log loss']:.5f}"
      f" (trial {int(top.trial)})")
chosen = list(best(trials).items())
for part in (chosen[:3], chosen[3:]):
    print("  " + ", ".join(f"{k} {v:.3g}" for k, v in part))

full = search(table, Budget(BUDGET.trials, BUDGET.trials))
end = full.loc[full["log loss"].idxmin()]
print(f"\nThe same search run to {len(full)} trials:"
      f" best {end['log loss']:.5f} (trial {int(end.trial)})")
stop = len(trials)
gain = full["best so far"][stop - 1] - end["log loss"]
print(f"  trials {stop + 1}-{len(full)} improved on the best of the"
      f" first {stop} by {gain:.5f}")
