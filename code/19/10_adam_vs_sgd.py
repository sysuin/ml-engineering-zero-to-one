# Why Adam gets further than SGD here: gradients of every size.
import pandas as pd
import torch

from foresight.config import SEED, seed_everything
from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import features
from foresight.models.mlp import (epoch, known_split, log_loss_of,
                                  network)

seed_everything()
train, _ = known_split(pd.read_parquet(TABLE))
X = Standardiser().fit(features(train).to_numpy()).transform(
    features(train).to_numpy())
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(train.not_renewed.to_numpy(), dtype=torch.float32)


def fresh():
    torch.manual_seed(SEED)
    return network(16, hidden=(32, 32), dropout=0.0), \
        torch.Generator().manual_seed(SEED)


# How big is each layer's gradient over the first epoch, from the start?
net, g = fresh()
sums = [torch.zeros_like(p) for p in net.parameters()]
batches = torch.randperm(len(y), generator=g).split(128)
for idx in batches:
    net.zero_grad()
    torch.nn.BCEWithLogitsLoss()(net(X[idx]).squeeze(1),
                                 y[idx]).backward()
    for s, p in zip(sums, net.parameters()):
        s += p.grad ** 2
names = ["layer 1 weights", "layer 1 biases", "layer 2 weights",
         "layer 2 biases", "output weights", "output bias"]
print("Gradient size over the first epoch (root mean square)")
print(f"  {'':18}{'count':>6}{'size':>10}")
for name, s in zip(names, sums):
    rms = (s / len(batches)).mean().sqrt().item()
    print(f"  {name:<18}{s.numel():>6}{rms:>10.4f}")

print(f"\n{'training log loss after epoch':<30}{10:>7}{30:>7}{60:>7}")
for Opt, lrs in ((torch.optim.SGD, (0.01, 0.1, 0.3, 1.0)),
                 (torch.optim.Adam, (0.0003, 0.001, 0.003))):
    for lr in lrs:
        net, g = fresh()
        opt = Opt(net.parameters(), lr=lr)
        at = {}
        for e in range(1, 61):
            epoch(net, opt, X, y, 128, g)
            if e in (10, 30, 60):
                at[e] = log_loss_of(net, X, y)
        name = f"{Opt.__name__} {lr:g}"
        print(f"{name:<30}"
              + "".join(f"{v:>7.4f}" for v in at.values()))
