# Early stopping: Chapter 7's descent, watched on rows it does not fit.
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from foresight.data.build_table import TABLE
from foresight.evaluate import auc
from foresight.models.linear import Standardiser, descend
from foresight.models.logistic import (features, log_loss,
                                       log_loss_gradient, sigmoid)

table = pd.read_parquet(TABLE)
# Both inside the training split. Every outcome fitted on was known
# (1 January 2024) before the first watched cohort's mark (31 January).
fit = table[table.end_date.between("2023-01-01", "2023-12-31")]
watch = table[table.end_date.between("2024-04-01", "2024-06-30")]
y, yw = fit.not_renewed.to_numpy(), watch.not_renewed.to_numpy()
print(f"fit on {len(fit):,} contracts; watch {len(watch):,}"
      f" ({yw.sum()} leavers)")

scale = Standardiser().fit(features(fit).to_numpy())
poly = PolynomialFeatures(2, include_bias=False)   # the 152 columns
A = poly.fit_transform(scale.transform(features(fit).to_numpy()))
B = poly.transform(scale.transform(features(watch).to_numpy()))
z = Standardiser().fit(A)
A, B = z.transform(A), z.transform(B)

steps = 20_000
w, b, path = descend(A, y, 0.5, steps, gradient=log_loss_gradient,
                     path=True)
path.append((w, b))


def losses(t):
    wt, bt = path[t]
    with np.errstate(over="ignore"):    # exp of a huge z: p is 0 or 1
        return (log_loss(y, sigmoid(A @ wt + bt)),
                log_loss(yw, sigmoid(B @ wt + bt)))


curve = [(t, *losses(t)) for t in range(0, steps + 1, 10)]
best = min(curve, key=lambda r: r[2])[0]
print(f"{'step':>8}{'fit loss':>10}{'watched loss':>14}"
      f"{'watched AUC':>13}{'size of w':>11}")
for t in sorted({0, 10, 30, best, 300, 1_000, 5_000, steps}):
    wt = path[t][0]
    mark = "  lowest" if t == best else ""
    _, fit_loss, watched = curve[t // 10]
    print(f"{t:>8,}{fit_loss:>10.4f}{watched:>14.4f}"
          f"{auc(yw, B @ wt):>13.4f}{np.sqrt(wt @ wt):>11.2f}{mark}")

print("\nthe ridge penalty instead, on the same rows")
for s in (0.0001, 0.001, 0.01, 0.03, 0.1):
    m = LogisticRegression(C=1 / (2 * s * len(y)), tol=1e-8,
                           max_iter=10_000).fit(A, y)
    p, wt = m.predict_proba(B)[:, 1], m.coef_[0]
    print(f"  strength {s:<8g}{log_loss(yw, p):>14.4f}"
          f"{auc(yw, p):>13.4f}{np.sqrt(wt @ wt):>11.2f}")

F = scale.transform(features(fit).to_numpy())
G = scale.transform(features(watch).to_numpy())
w16, b16 = descend(F, y, 0.5, steps, gradient=log_loss_gradient)
p16 = sigmoid(G @ w16 + b16)
print(f"  {'16 columns':<17}{log_loss(yw, p16):>14.4f}"
      f"{auc(yw, p16):>13.4f}{np.sqrt(w16 @ w16):>11.2f}")

with open("code/09/09_early_stopping.json", "w") as f:
    json.dump({"curve": curve, "best": best}, f)
