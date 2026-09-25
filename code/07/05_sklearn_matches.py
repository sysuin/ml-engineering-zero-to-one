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

print(f"{'per standard deviation':<24}{'descent':>9}{'sklearn':>9}"
      f"{'odds x':>8}")
rows = [("intercept", ours["b"], lib.intercept_[0])]
rows += list(zip(ours["columns"], ours["w"], lib.coef_[0]))
for name, a, s in rows:
    odds = "" if name == "intercept" else f"{np.exp(s):>8.2f}"
    print(f"{name:<24}{a:>+9.4f}{s:>+9.4f}{odds}")
gaps = [abs(a - s) for _, a, s in rows]
print(f"Largest difference: {max(gaps):.1e}")
