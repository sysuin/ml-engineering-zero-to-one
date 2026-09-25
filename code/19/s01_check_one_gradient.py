# Exercise 1: one gradient by nudging, and a unit that fires.
import json

import numpy as np

d = json.load(open("code/19/04_forward_by_hand.json"))
x, y = np.array(d["x"]), d["y"]


def loss(W1, b1, v, c):
    h = np.maximum(x @ W1 + b1, 0)
    return -np.log(1 / (1 + np.exp(-(h @ v + c))))    # y = 1


W1, b1 = np.array(d["W1"]), np.array(d["b1"])
v, c = np.array(d["v"]), d["c"]
for (i, j), name in (((0, 0), "w11"), ((0, 1), "w12")):
    up, down = W1.copy(), W1.copy()
    up[i, j] += 0.001
    down[i, j] -= 0.001
    slope = (loss(up, b1, v, c) - loss(down, b1, v, c)) / 0.002
    print(f"{name}: nudging gives {slope:+.4f}")

# Make unit 2 fire for this contract: its sum was -0.37.
b1 = np.array([d["b1"][0], 0.5])
z1 = x @ W1 + b1
p = 1 / (1 + np.exp(-(np.maximum(z1, 0) @ v + c)))
dz = p - y
dz1 = dz * v * (z1 > 0)
print(f"\nWith b2 = +0.5, unit 2's sum is {z1[1]:+.2f}")
print(f"  v2 gradient {dz * max(z1[1], 0):+.4f}")
print(f"  w12 gradient {dz1[1] * x[0]:+.4f},"
      f" w22 {dz1[1] * x[1]:+.4f}, b2 {dz1[1]:+.4f}")
