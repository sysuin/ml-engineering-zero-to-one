"""
A small neural network for renewal risk, in PyTorch. Chapter 19 writes
it, to test the network against gradient boosting and the rule on
Chapter 8's backtest.

    python -m foresight.models.mlp          the backtest, validation

RenewalMLP has the same fit(rows) and predict_proba(rows) as Chapter 7's
RenewalRisk, so evaluate(make_model=RenewalMLP) runs it unchanged. It
sees Chapter 7's sixteen columns unless given others, standardised on
its training rows.

How many epochs to train is chosen inside fit(), from the training rows
alone: the latest months of known outcomes are held back, the network
trains on the rest while log loss on the held-back months is watched,
and the epoch where that loss was lowest is kept. The network is then
trained again, from the same starting weights, on every row for that
many epochs. Nothing from the rows being scored is used to choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from foresight.config import SEED
from foresight.models.linear import Standardiser
from foresight.models.logistic import VALIDATION, features


def network(inputs: int, hidden=(32, 32), dropout: float = 0.1):
    """Layers of ReLU units, then one output: the log-odds."""
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.ReLU(),
                   nn.Dropout(dropout)]
        inputs = width
    layers.append(nn.Linear(inputs, 1))
    return nn.Sequential(*layers)


def epoch(net, opt, X, y, batch: int, g) -> float:
    """One pass over the rows in a shuffled order, a step per batch.
    Returns the mean training log loss over the pass."""
    net.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for idx in torch.randperm(len(y), generator=g).split(batch):
        opt.zero_grad()
        loss = loss_fn(net(X[idx]).squeeze(1), y[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(y)


def log_loss_of(net, X, y) -> float:
    """Log loss with dropout switched off, as the model will be used."""
    net.eval()
    with torch.no_grad():
        return nn.BCEWithLogitsLoss()(net(X).squeeze(1), y).item()


class RenewalMLP:
    """Standardise, then a network trained by Adam on log loss."""

    def __init__(self, hidden=(32, 32), dropout: float = 0.1,
                 weight_decay: float = 1e-4, lr: float = 1e-3,
                 batch: int = 128, max_epochs: int = 100,
                 patience: int = 10, holdout_months: int = 3,
                 seed: int = SEED, columns=features):
        self.hidden, self.dropout = tuple(hidden), dropout
        self.weight_decay, self.lr, self.batch = weight_decay, lr, batch
        self.max_epochs, self.patience = max_epochs, patience
        self.holdout_months, self.seed = holdout_months, seed
        self.columns = columns          # rows -> a frame of numbers

    # -------------------------------------------------- the pieces
    def _tensors(self, rows: pd.DataFrame):
        X = self.scaler_.transform(self.columns(rows).to_numpy())
        y = rows.not_renewed.to_numpy()
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    def _fresh(self, inputs: int):
        """The same starting weights and batch order on every call."""
        torch.manual_seed(self.seed)
        net = network(inputs, self.hidden, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        return net, opt, torch.Generator().manual_seed(self.seed)

    def choose_epochs(self, rows: pd.DataFrame) -> int:
        """Train on all but the latest months, watch log loss on them,
        and return the epoch where it was lowest."""
        marks = np.sort(rows.moment.unique())
        held = rows.moment >= marks[-self.holdout_months]
        X, y = self._tensors(rows[~held])
        Xh, yh = self._tensors(rows[held])
        net, opt, g = self._fresh(X.shape[1])
        self.history_ = []                  # (train, held-back) losses
        best, best_epoch = np.inf, 1
        for e in range(1, self.max_epochs + 1):
            train = epoch(net, opt, X, y, self.batch, g)
            held_loss = log_loss_of(net, Xh, yh)
            self.history_.append((train, held_loss))
            if held_loss < best:
                best, best_epoch = held_loss, e
            elif e - best_epoch >= self.patience:
                break
        return best_epoch

    # -------------------------------------------------- the interface
    def fit(self, rows: pd.DataFrame):
        self.scaler_ = Standardiser().fit(self.columns(rows).to_numpy())
        self.epochs_ = self.choose_epochs(rows)
        X, y = self._tensors(rows)
        self.net_, opt, g = self._fresh(X.shape[1])
        for _ in range(self.epochs_):
            epoch(self.net_, opt, X, y, self.batch, g)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Each contract's chance of not renewing."""
        X, _ = self._tensors(rows)
        self.net_.eval()
        with torch.no_grad():
            z = self.net_(X).squeeze(1)
        return torch.sigmoid(z).numpy().astype(float)


def known_split(table: pd.DataFrame):
    """Chapter 19's fixed split: the validation contracts, and the
    contracts whose outcome was known on the first validation mark."""
    valid = table[table.end_date.between(*VALIDATION)]
    first_mark = valid.moment.min()
    train = table[(table.end_date >= "2023-01-01")
                  & (table.end_date < first_mark)]
    return (train.reset_index(drop=True),
            valid.reset_index(drop=True))


def main() -> None:
    from foresight.config import seed_everything
    from foresight.evaluate import evaluate, page
    seed_everything()
    print(page(evaluate(make_model=RenewalMLP),
               title="Foresight evaluation: the network"))


if __name__ == "__main__":
    main()
