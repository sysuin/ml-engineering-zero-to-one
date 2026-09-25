# Check backpropagation against the slope measured by nudging.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.models.linear import Standardiser
from foresight.models.logistic import features, log_loss, sigmoid
from foresight.models.mlp import known_split

train, _ = known_split(pd.read_parquet(TABLE))
X, y = features(train).to_numpy(), train.not_renewed.to_numpy()
Z = Standardiser().fit(X).transform(X)
start = json.load(open("code/19/06_numpy_network.json"))["start"]
theta = np.concatenate([np.ravel(start[k]) for k in ("W1", "b1", "v")]
                       + [[start["c"]]])       # all 145 in one row


def unpack(t):
    return t[:128].reshape(16, 8), t[128:136], t[136:144], t[144]


def logit(t):
    W1, b1, v, c = unpack(t)
    return np.maximum(Z @ W1 + b1, 0) @ v + c


def loss(t):
    """Log loss worked out from the logit z, never from p."""
    z = logit(t)
    return np.mean(np.logaddexp(0, z) - y * z)


def loss_from_p(t):
    """Chapter 7's log_loss(), which works from p and log(1 - p)."""
    return log_loss(y, sigmoid(logit(t)))


def backprop(t, relu_mask=True):
    W1, b1, v, c = unpack(t)
    z1 = Z @ W1 + b1
    h = np.maximum(z1, 0)
    dz = (sigmoid(h @ v + c) - y) / len(y)
    dz1 = np.outer(dz, v) * ((z1 > 0) if relu_mask else 1)
    return np.concatenate([(Z.T @ dz1).ravel(), dz1.sum(axis=0),
                           h.T @ dz, [dz.sum()]])


def nudge(f, eps=1e-6):
    """Each parameter's slope: nudge it up and down, and measure."""
    return np.array([(f(theta + eps * e) - f(theta - eps * e))
                     / (2 * eps) for e in np.eye(len(theta))])


def worst(a, b):
    """The largest relative gap between two sets of gradients."""
    return (np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b),
                                       1e-12)).max()


grad, nudged = backprop(theta), nudge(loss)
print(f"{'parameter':<12}{'backprop':>13}{'nudging':>13}")
for i, name in ((0, "W1[0, 0]"), (128, "b1[0]"), (136, "v[0]"),
                (144, "c")):
    print(f"{name:<12}{grad[i]:>+13.8f}{nudged[i]:>+13.8f}")
print(f"\nLargest relative gap, all {len(theta)} parameters")
print(f"  backprop as written            {worst(grad, nudged):.1e}")
bad = backprop(theta, relu_mask=False)
print(f"  ReLU's mask left out           {worst(bad, nudged):.1e}")
print(f"  nudging the loss from p        "
      f"{worst(grad, nudge(loss_from_p)):.1e}")
p = sigmoid(logit(theta))
sure = np.minimum(p, 1 - p) < 1e-6
print(f"Rows given p within a millionth of 0 or 1: {sure.sum()}")
row = train[sure].iloc[0]
print(f"  contract {row.contract_id}, {row.segment}, {row.orders_90d}"
      f" orders in 90 days: {Z[sure][0, 1]:.1f} sd out")
Z, y = Z[~sure], y[~sure]
print("Without that row, nudging the loss from p: "
      f"{worst(backprop(theta), nudge(loss_from_p)):.1e}")
