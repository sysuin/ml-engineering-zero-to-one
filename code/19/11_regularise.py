# A network big enough to overfit, and three ways to stop it.
import json

import pandas as pd
import torch

from foresight.config import SEED, seed_everything
from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import at_capacity, features
from foresight.models.mlp import (RenewalMLP, epoch, known_split,
                                  log_loss_of, network)

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
EPOCHS = 150
runs = {"no regularisation": (0.0, 0.0),     # (dropout, decay)
        "dropout 0.3": (0.3, 0.0), "dropout 0.5": (0.5, 0.0),
        "weight decay 0.1": (0.0, 0.1), "weight decay 1": (0.0, 1.0),
        "dropout and decay": (0.3, 1.0)}
curves = {}
print(f"{'64 -> 64, Adam 0.001':<22}{'lowest valid loss':>18}"
      f"{'at epoch ' + str(EPOCHS):>16}{'leavers':>9}")
for name, (drop, decay) in runs.items():
    torch.manual_seed(SEED)
    net = network(16, hidden=(64, 64), dropout=drop)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3,
                            weight_decay=decay)
    g = torch.Generator().manual_seed(SEED)
    curve = []
    for e in range(EPOCHS):
        epoch(net, opt, X, y, 128, g)
        curve.append((log_loss_of(net, X, y), log_loss_of(net, Xv, yv)))
    curves[name] = curve
    best = min(range(EPOCHS), key=lambda e: curve[e][1])
    with torch.no_grad():
        p = torch.sigmoid(net(Xv).squeeze(1)).numpy()
    hits = at_capacity(valid, p)["leavers"]
    print(f"{name:<22}{curve[best][1]:>11.4f} (ep {best + 1:>3})"
          f"{curve[-1][1]:>16.4f}{hits:>9}")

# Early stopping without looking at validation: RenewalMLP holds back
# the latest three months of the training rows and watches those.
stop = RenewalMLP(hidden=(64, 64), dropout=0.0, weight_decay=0.0,
                  max_epochs=EPOCHS, patience=EPOCHS).fit(train)
held = [h for _, h in stop.history_]
p = stop.predict_proba(valid)
print(f"\nEarly stopping on the held-back months: epoch {stop.epochs_}")
print(f"  validation loss {log_loss_of(stop.net_, Xv, yv):.4f},"
      f" leavers in the top 40s {at_capacity(valid, p)['leavers']}")

with open("code/19/11_regularise.json", "w") as f:
    json.dump({"curves": curves, "held": held,
               "chosen": stop.epochs_}, f)
