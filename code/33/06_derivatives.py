# Slopes by nudging and by rule, one descent, and the chain rule.
import numpy as np

from foresight.models.logistic import (TRAIN, RenewalRisk, features,
                                       load, log_loss, sigmoid)

train = load(*TRAIN)
y = train.not_renewed.to_numpy()


def loss(b):
    """Log loss when every contract gets the same score, sigmoid(b)."""
    return log_loss(y, np.full(len(y), sigmoid(b)))


# The slope at b = -1, by nudging, against the rule mean(p - y).
b = -1.0
print(f"Slope of the loss at b = {b}")
for h in (1.0, 0.1, 0.01, 0.001):
    print(f"  nudge {h:<6} (L(b + h) - L(b)) / h ="
          f" {(loss(b + h) - loss(b)) / h:.6f}")
rule = np.mean(sigmoid(b) - y)
print(f"  by the rule, mean(p - y)      = {rule:.6f}")

# Walk downhill: b <- b - rate x slope.
b, rate = 0.0, 2.0
print("\nDescent on b alone, learning rate 2")
for step in range(31):
    if step in (0, 1, 2, 5, 10, 30):
        print(f"  step {step:>2}  b = {b:+.4f}  loss {loss(b):.6f}")
    b -= rate * np.mean(sigmoid(b) - y)
print(f"  the bottom: log-odds of the rate,"
      f" {np.log(y.mean() / (1 - y.mean())):+.4f}")

# The chain rule on one contract of the fitted model:
# d loss / d w_j = (d loss / d z) (d z / d w_j) = (p - y) x_j
model = RenewalRisk().fit(train)
w, c = model.model_.coef_[0], model.model_.intercept_[0]
Z = model.scaler_.transform(features(train).to_numpy())
i = int(np.argmax(Z @ w))
j = model.columns_.index("days_since_order")
x, target = Z[i], y[i:i + 1]


def one(wj):
    v = w.copy()
    v[j] = wj
    return log_loss(target, sigmoid(np.array([v @ x + c])))


p = sigmoid(w @ x + c)
h = 1e-6
print(f"\nContract {train.contract_id[i]}: p = {p:.4f}, y = {y[i]}")
print(f"  (p - y) x_j = ({p:.4f} - {y[i]}) x {x[j]:.3f}"
      f" = {(p - y[i]) * x[j]:.5f}")
nudged = (one(w[j] + h) - one(w[j] - h)) / (2 * h)
print(f"  by nudging w_j both ways: {nudged:.5f}")
