# Exercise 2: the network with heavy-tailed columns on a log scale.
# timeout: 300
import numpy as np
import pandas as pd

from foresight.config import SEED, seed_everything
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.logistic import features
from foresight.models.mlp import RenewalMLP

seed_everything()
table = pd.read_parquet(TABLE)
HEAVY = ["days_since_order", "orders_90d", "orders_prev_90d",
         "tickets_90d"]


def logged(rows):
    """Chapter 7's columns, the four heavy-tailed ones as log(1 + x)."""
    X = features(rows)
    X[HEAVY] = np.log1p(X[HEAVY])
    return X


print(f"{'seed':<10}{'as Chapter 7':>18}{'logged':>18}")
print(f"{'':<10}{'leavers   AUC':>18}{'leavers   AUC':>18}")
for seed in (SEED, SEED + 1, SEED + 2):
    cells = []
    for cols in (features, logged):
        m = measure(backtest(table, *SPLITS["validation"],
                             lambda: RenewalMLP(seed=seed,
                                                columns=cols)))
        p, a = m["precision"]["model"], m["auc"]["model"]
        cells.append(f"{p * 240:>10.0f}{a:>8.3f}")
    print(f"{seed:<10}" + "".join(cells))
