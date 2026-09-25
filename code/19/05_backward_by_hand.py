# Backpropagation on the same contract: blame, passed backwards.
import json

import numpy as np

d = json.load(open("code/19/04_forward_by_hand.json"))
x, W1, b1 = np.array(d["x"]), np.array(d["W1"]), np.array(d["b1"])
v, c, y = np.array(d["v"]), d["c"], d["y"]


def forward(W1, b1, v, c):
    z1 = x @ W1 + b1
    h = np.maximum(z1, 0)
    p = 1 / (1 + np.exp(-(h @ v + c)))
    return z1, h, p


z1, h, p = forward(W1, b1, v, c)
dz = p - y                          # the output's blame (Chapter 7)
dv, dc = dz * h, dz                 # each output weight's share
dh = dz * v                         # blame sent back to each unit
dz1 = dh * (z1 > 0)                 # ReLU passes it only if it fired
dW1, db1 = np.outer(x, dz1), dz1    # each first-layer weight's share
print(f"output blame  p - y = {p:.4f} - {y} = {dz:+.4f}")
for j in range(2):
    state = "fired" if z1[j] > 0 else "silent"
    print(f"unit {j + 1} ({state}): blame arriving {dh[j]:+.4f},"
          f" passed on {dz1[j] + 0:+.4f}")
print(f"\n{'parameter':<14}{'value':>7}{'gradient':>10}{'after':>8}")
grads = [("v1", v[0], dv[0]), ("v2", v[1], dv[1]), ("c", c, dc)]
for j in range(2):
    grads += [(f"w1{j + 1} (x1)", W1[0, j], dW1[0, j]),
              (f"w2{j + 1} (x2)", W1[1, j], dW1[1, j]),
              (f"b{j + 1}", b1[j], db1[j])]
lr = 0.5
for name, value, grad in grads:
    print(f"{name:<14}{value:>+7.2f}{grad + 0:>+10.4f}"
          f"{value - lr * grad:>+8.3f}")

new = forward(W1 - lr * dW1, b1 - lr * db1, v - lr * dv, c - lr * dc)
print(f"\nOne step at learning rate {lr}")
print(f"  chance of not renewing {p:.4f} -> {new[2]:.4f}")
print(f"  log loss {-np.log(p):.4f} -> {-np.log(new[2]):.4f}")

with open("code/19/05_backward_by_hand.json", "w") as f:
    json.dump({"z1": z1.tolist(), "h": h.tolist(), "p": p, "dz": dz,
               "dh": dh.tolist(), "dz1": dz1.tolist(),
               "grads": [(n, a, g) for n, a, g in grads]}, f)
