# The same model from scikit-learn, weight by weight.
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from foresight.models.linear import Standardiser
from foresight.models.logistic import TRAIN, features, load

train = load(*TRAIN)
X, y = features(train), train.not_renewed.to_numpy()
Z = Standardiser().fit(X.to_numpy()).transform(X.to_numpy())
ours = json.load(open("code/07/04_logistic_numpy.json"))

# C=np.inf: no penalty on the weights, the loss exactly as in §7.4.
lib = LogisticRegression(C=np.inf, tol=1e-10,
                         max_iter=10_000).fit(Z, y)

# sd: one standard deviation in the column's own units. For a yes-or-no
# column, "0 to 1" is the odds multiplier for the step from 0 to 1.
sd = X.std(ddof=0)
print(f"{'per standard deviation':<24}{'descent':>9}{'sklearn':>9}"
      f"{'odds x':>8}{'sd':>9}{'0 to 1':>8}")
rows = [("intercept", ours["b"], lib.intercept_[0])]
rows += list(zip(ours["columns"], ours["w"], lib.coef_[0]))
for name, a, s in rows:
    line = f"{name:<24}{a:>+9.4f}{s:>+9.4f}"
    if name != "intercept":
        line += f"{np.exp(s):>8.2f}{sd[name]:>9.2f}"
        if set(X[name].unique()) <= {0.0, 1.0}:
            line += f"{np.exp(s / sd[name]):>8.2f}"
    print(line)
gaps = [abs(a - s) for _, a, s in rows]
print(f"Largest difference: {max(gaps):.1e}")
