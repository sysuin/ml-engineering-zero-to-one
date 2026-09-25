# The network's result, trained from five different starting points.
# timeout: 300
import numpy as np
import pandas as pd

from foresight.config import SEED, seed_everything
from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, backtest, measure
from foresight.models.mlp import RenewalMLP

seed_everything()
table = pd.read_parquet(TABLE)
print(f"{'network seed':<16}{'leavers':>8}{'precision':>11}{'AUC':>8}"
      f"{'epochs chosen':>22}")
results = []
for seed in (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4):
    fitted = []

    def make(seed=seed):
        model = RenewalMLP(seed=seed)
        fitted.append(model)
        return model

    m = measure(backtest(table, *SPLITS["validation"], make))
    p, a = m["precision"]["model"], m["auc"]["model"]
    results.append((p, a))
    chosen = " ".join(f"{f.epochs_:>2}" for f in fitted)
    print(f"{seed:<16}{p * 240:>8.0f}{p:>11.1%}{a:>8.3f}{chosen:>22}")
p, a = np.array(results).T
print(f"Range: {p.min() * 240:.0f} to {p.max() * 240:.0f} leavers,"
      f" AUC {a.min():.3f} to {a.max():.3f}")
