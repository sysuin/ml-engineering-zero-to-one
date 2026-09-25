# Batches, epochs and optimisers: three ways to spend 30 passes.
import pandas as pd
import torch

from foresight.config import SEED, seed_everything
from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import features
from foresight.models.mlp import (epoch, known_split, log_loss_of,
                                  network)

seed_everything()
train, valid = known_split(pd.read_parquet(TABLE))
scaler = Standardiser().fit(features(train).to_numpy())


def tensors(rows):
    X = scaler.transform(features(rows).to_numpy())
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(rows.not_renewed.to_numpy(),
                         dtype=torch.float32))


X, y = tensors(train)
Xv, yv = tensors(valid)
runs = [("full batch, SGD 0.1", len(y), torch.optim.SGD, 0.1),
        ("128s, SGD 0.1", 128, torch.optim.SGD, 0.1),
        ("128s, Adam 0.001", 128, torch.optim.Adam, 1e-3)]
show = (1, 2, 5, 10, 20, 30)
print(f"{'loss after epoch':<22}"
      + "".join(f"{e:>6}" for e in show) + f"{'valid':>7}")
for name, batch, Opt, lr in runs:
    torch.manual_seed(SEED)                 # the same starting weights
    net = network(16, hidden=(32, 32), dropout=0.0)
    opt, g = Opt(net.parameters(), lr=lr), torch.Generator()
    g.manual_seed(SEED)                     # the same shuffles
    losses = [epoch(net, opt, X, y, batch, g) for _ in range(30)]
    print(f"{name:<22}" + "".join(f"{losses[e - 1]:>6.3f}"
                                  for e in show)
          + f"{log_loss_of(net, Xv, yv):>7.3f}")
steps = -(-len(y) // 128)
print(f"\nOne epoch: {len(y):,} rows; {steps} steps in batches of 128,"
      " 1 in a full batch")
