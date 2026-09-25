# Inertia, silhouette and stability for k from 2 to 10, side by side.
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from foresight.config import SEED
from foresight.segments import (SEGMENT_DATE, account_features, load,
                                model_matrix)

orders, accounts = load()
f = account_features(SEGMENT_DATE, orders, accounts)
X = model_matrix(f)
X = (X - X.mean(0)) / X.std(0)

rows = []
print(f"{'k':>3}{'inertia':>10}{'drop':>8}{'silhouette':>12}"
      f"{'stability':>11}   smallest")
for k in range(2, 11):
    runs = [KMeans(k, n_init=10, random_state=SEED + i).fit(X)
            for i in range(5)]
    km = runs[0]
    sil = silhouette_score(X, km.labels_)
    stab = np.mean([adjusted_rand_score(km.labels_, r.labels_)
                    for r in runs[1:]])
    drop = f"{rows[-1]['inertia'] - km.inertia_:.0f}" if rows else ""
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil,
                 "stability": stab})
    small = np.bincount(km.labels_).min() / len(X)
    print(f"{k:>3}{km.inertia_:>10.0f}{drop:>8}{sil:>12.3f}"
          f"{stab:>11.2f}{small:>11.1%}")

# The elbow, found by rule: the k farthest below the straight line
# from the first point to the last (both axes scaled to 0-1).
ks = np.array([r["k"] for r in rows], dtype=float)
inert = np.array([r["inertia"] for r in rows])
x = (ks - ks[0]) / (ks[-1] - ks[0])
y = (inert - inert[-1]) / (inert[0] - inert[-1])
elbow = int(ks[np.argmax((1 - x) - y)])
best = max(rows, key=lambda r: r["silhouette"])
print(f"\nelbow by rule: k = {elbow}; highest silhouette: "
      f"k = {best['k']} ({best['silhouette']:.3f})")
json.dump({"rows": rows, "elbow": elbow, "best_silhouette": best["k"]},
          open("code/18/03_choose_k.json", "w"))
