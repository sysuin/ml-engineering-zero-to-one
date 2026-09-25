# The same network in PyTorch: same start, same data, same steps.
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from foresight.config import seed_everything
from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import features
from foresight.models.mlp import known_split

seed_everything()
train, _ = known_split(pd.read_parquet(TABLE))
X = features(train).to_numpy()
Z = torch.tensor(Standardiser().fit(X).transform(X))   # float64
y = torch.tensor(train.not_renewed.to_numpy(), dtype=torch.float64)
ours = json.load(open("code/19/06_numpy_network.json"))

net = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
net = net.double()
with torch.no_grad():                   # NumPy's starting weights
    net[0].weight.copy_(torch.tensor(ours["start"]["W1"]).T)
    net[0].bias.copy_(torch.tensor(ours["start"]["b1"]))
    net[2].weight.copy_(torch.tensor([ours["start"]["v"]]))
    net[2].bias.fill_(ours["start"]["c"])

loss_fn = nn.BCEWithLogitsLoss()        # sigmoid and log loss in one
opt = torch.optim.SGD(net.parameters(), lr=ours["lr"])
for step in range(ours["steps"]):
    opt.zero_grad()                     # forget the last gradients
    loss = loss_fn(net(Z).squeeze(1), y)
    loss.backward()                     # backpropagation, done for us
    opt.step()                          # w <- w - lr * gradient

end = ours["end"]
pairs = [("W1", net[0].weight.T, end["W1"]),
         ("b1", net[0].bias, end["b1"]),
         ("v", net[2].weight[0], end["v"]),
         ("c", net[2].bias, [end["c"]])]
print(f"After {ours['steps']:,} steps at learning rate {ours['lr']}")
print(f"  {'':4}{'numbers':>8}{'largest difference':>21}")
for name, torch_w, numpy_w in pairs:
    gap = np.abs(torch_w.detach().numpy() - np.array(numpy_w)).max()
    print(f"  {name:<4}{torch_w.numel():>8}{gap:>21.1e}")
print(f"  training log loss {loss_fn(net(Z).squeeze(1), y).item():.4f}")
