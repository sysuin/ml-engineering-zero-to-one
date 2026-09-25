# Two columns, two leashes: where each penalty stops the weights.
import json

import numpy as np
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, known_by
from foresight.models.logistic import TRAIN, log_loss, sigmoid
from foresight.models.regularised import RegularisedRisk

table = pd.read_parquet(TABLE)
valid = table[table.end_date.between(*SPLITS["validation"])]
train = known_by(table[table.end_date.between(*TRAIN)],
                 valid.moment.min())
keep = ("days_since_order", "tenure_days")


class TwoColumns(RegularisedRisk):
    def columns(self, rows):
        return super().columns(rows)[list(keep)]


print(f"{'':<16}{keep[0]:>18}{keep[1]:>13}"
      f"{'sum |w|':>9}{'sum w^2':>9}")
fits = {}
for name, pen, s in (("unpenalised", "l2", 0.0), ("ridge 0.01", "l2",
                     0.01), ("lasso 0.01", "l1", 0.01)):
    m = fits[name] = TwoColumns(pen, s).fit(train)
    w = m.weights().to_numpy()
    print(f"{name:<16}{w[0]:>18.3f}{w[1]:>13.3f}"
          f"{np.abs(w).sum():>9.3f}{(w ** 2).sum():>9.3f}")

# For the figure's contour lines: the log loss over a grid of the two
# weights, each point with its own best intercept (Newton's method).
free = fits["unpenalised"]
Z = free.scaler_.transform(free.columns(train).to_numpy())
y = train.not_renewed.to_numpy()
w1, w2 = np.linspace(-0.1, 0.7, 81), np.linspace(-0.4, 0.4, 81)
grid = []
for c in w2:
    s = Z @ np.array([w1, np.full_like(w1, c)])   # rows x 81 points
    b = np.full(len(w1), np.log(y.mean() / (1 - y.mean())))
    for _ in range(20):
        p = sigmoid(s + b)
        b -= (p - y[:, None]).mean(0) / (p * (1 - p)).mean(0)
    grid.append([log_loss(y, sigmoid(s[:, i] + b[i]))
                 for i in range(len(w1))])
with open("code/09/11_two_weights.json", "w") as f:
    json.dump({"w1": w1.tolist(), "w2": w2.tolist(), "loss": grid,
               "fits": {n: m.weights().tolist()
                        for n, m in fits.items()}}, f)
