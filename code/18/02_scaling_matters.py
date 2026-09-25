# One set of features clustered three ways: raw, logged, standardised.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from foresight.config import SEED
from foresight.segments import (RAW_COLUMNS, SEGMENT_DATE, Segmenter,
                                account_features, load, model_matrix)

orders, accounts = load()
f = account_features(SEGMENT_DATE, orders, accounts)
logged = model_matrix(f)
versions = {
    "raw units": f[RAW_COLUMNS].to_numpy(),
    "logs, unscaled": logged,
    "standardised": (logged - logged.mean(0)) / logged.std(0),
}

short = ["orders", "basket", "recency", "tenure", "trend"]
print(f"{'standard deviation':22}" + "".join(f"{c:>9}" for c in short))
for name, X in list(versions.items())[:2]:
    print(f"{name:22}" + "".join(f"{s:>9.2f}" for s in X.std(0)))

final = Segmenter(4).fit(f).predict(f)
print(f"\n{'k = 4':16}{'sizes':>22}{'agree':>7}  differ most in")
for name, X in versions.items():
    labels = KMeans(4, n_init=10, random_state=SEED).fit_predict(X)
    sizes = np.sort(np.bincount(labels))[::-1]
    agree = adjusted_rand_score(final, labels)
    med = f.assign(s=labels).groupby("s")[RAW_COLUMNS].median()
    spread = (med.max() - med.min()) / f[RAW_COLUMNS].std().values
    top = spread.sort_values(ascending=False).index[:2]
    names = ", ".join(short[RAW_COLUMNS.index(c)] for c in top)
    print(f"{name:16}{str(sizes):>22}{agree:>7.2f}  {names}")
