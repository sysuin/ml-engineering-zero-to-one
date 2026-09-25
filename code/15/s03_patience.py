# Exercise 3: where patience of 5, 10, 20 and 40 trials would have
# stopped the sixty-trial search, and what each would have chosen.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.tune import BUDGET, NOISE, Budget, search

table = pd.read_parquet(TABLE)
full = search(table, Budget(BUDGET.trials, BUDGET.trials))
losses = full["log loss"].tolist()

print(f"{'patience':>8}{'stops at':>10}{'best':>10}{'found at':>10}")
for patience in (5, 10, 20, 40):
    level, since, stop = float("inf"), 0, len(losses)
    for i, loss in enumerate(losses, start=1):
        if loss < level - NOISE:
            level, since = loss, 0
        else:
            level, since = min(level, loss), since + 1
        if since >= patience:
            stop = i
            break
    seen = full.iloc[:stop]
    at = int(seen.loc[seen["log loss"].idxmin(), "trial"])
    print(f"{patience:>8}{stop:>10}{seen['log loss'].min():>10.5f}"
          f"{at:>10}")
