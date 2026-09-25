# A layer of three units learns the shape one neuron could not.
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from foresight.config import SEED
from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
days = features(train)[["days_since_order"]].to_numpy()
y = train.not_renewed.to_numpy()
mu, sd = days.mean(), days.std()
x = (days - mu) / sd                         # standardised, as ever

one = LogisticRegression(C=np.inf).fit(x, y)
warnings.simplefilter("ignore", ConvergenceWarning)
net = MLPClassifier(hidden_layer_sizes=(3,), solver="lbfgs",
                    alpha=0, max_iter=5_000,
                    random_state=SEED).fit(x, y)

print(f"  {'days':<10}{'rows':>6}{'left':>8}{'one neuron':>12}"
      f"{'network':>9}")
bands = [(1, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 180),
         (181, 411)]
for lo, hi in bands:
    inside = (days[:, 0] >= lo) & (days[:, 0] <= hi)
    a = one.predict_proba(x[inside])[:, 1].mean()
    b = net.predict_proba(x[inside])[:, 1].mean()
    print(f"  {f'{lo}-{hi}':<10}{inside.sum():>6}"
          f"{y[inside].mean():>8.1%}{a:>12.1%}{b:>9.1%}")

# Each hidden unit is max(0, a * days + c): a hinge at one day.
W1, b1 = net.coefs_[0][0], net.intercepts_[0]
W2 = net.coefs_[1][:, 0]
print("\nWhat the three hidden units learned")
for j in range(3):
    hinge = mu - b1[j] / W1[j] * sd          # where a * x + c = 0
    side = "after" if W1[j] > 0 else "before"
    print(f"  unit {j + 1}: on {side} {hinge:.0f} days,"
          f" output weight {W2[j]:+.2f}")

grid = np.arange(0, 412)
g = ((grid - mu) / sd)[:, None]
with open("code/19/02_learned_features.json", "w") as f:
    json.dump({"days": grid.tolist(),
               "one": one.predict_proba(g)[:, 1].tolist(),
               "net": net.predict_proba(g)[:, 1].tolist()}, f)
