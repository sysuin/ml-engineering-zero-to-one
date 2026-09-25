# Exercise 5: how far one desk agrees with itself on priority.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from foresight.triage.evaluate import kappa
from foresight.triage.features import tickets, tokens

t = tickets()
south = t[(t.desk == "South desk")
          & (t.opened_at < "2025-01-01")].reset_index(drop=True)
vec = TfidfVectorizer(tokenizer=tokens, lowercase=False,
                      token_pattern=None).fit(south.body)
X = vec.transform(south.body)
nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
dist, idx = nn.kneighbors(X)
rows = []
for i in range(len(south)):
    for d, j in zip(dist[i, 1:], idx[i, 1:]):
        if 1 - d >= 0.9 and south.day[j] != south.day[i]:
            rows.append((south.priority[i], south.priority[j]))
            break
pairs = pd.DataFrame(rows, columns=["a", "b"])
print(f"{len(pairs):,} South tickets with a near twin from another day")
print(f"same priority {(pairs.a == pairs.b).mean():.1%}, kappa "
      f"{kappa(pairs.a, pairs.b):.3f}")
