# Word vectors learned from Meridian's tickets, then used as features.
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from foresight.config import rng, seed_everything
from foresight.triage.features import (pemberton, relabel, split,
                                       tickets, tokens)
from foresight.triage.model import TriageModel
from foresight.triage.neural import (Vocabulary, average_vectors,
                                     nearest, word_vectors)

seed_everything()
t = tickets()
t["category"] = relabel(t)
train, valid, _ = split(t)
words = [tokens(b) for b in train.body]
vocab = Vocabulary(words)
W = word_vectors(words, vocab, dim=32)          # no labels used
print(f"{len(vocab) - 2} words, 32 numbers each\n")
for w in ["stock", "twice", "crushed", "commande", "urgent"]:
    print(f"  {w:9}-> " + ", ".join(nearest(w, vocab, W)))

scale = StandardScaler().fit(average_vectors(words, vocab, W))
A = scale.transform(average_vectors(words, vocab, W))
B = scale.transform(average_vectors([tokens(b) for b in valid.body],
                                    vocab, W))
tfidf = TriageModel().fit(train.body, train.priority, train.category)
pred = {"TF-IDF": tfidf.predict(valid.body)}
pred["word vectors"] = {
    label: LogisticRegression(max_iter=2000).fit(A, train[label])
    .predict(B) for label in ["priority", "category"]}

cloth = pemberton(valid).to_numpy()
print(f"\nvalidation accuracy ({(~cloth).sum():,} ordinary tickets, "
      f"{cloth.sum():,} about the")
print("Pemberton cloth)")
print(f"{'':15}{'priority':>18}{'category':>20}")
print(f"{'':15}" + f"{'ordinary':>11}{'cloth':>8}" * 2)
scores = {}
for name, p in pred.items():
    cells = ""
    for label in ["priority", "category"]:
        hit = np.asarray(p[label]) == valid[label].to_numpy()
        scores[f"{name} {label}"] = [hit[~cloth].mean(),
                                     hit[cloth].mean()]
        cells += f"{hit[~cloth].mean():11.1%}{hit[cloth].mean():8.1%}"
    print(f"  {name:13}{cells}")

# Two dimensions of the ticket vectors, for the figure.
sample = rng().choice(len(train), 1500, replace=False)
centred = A[sample] - A[sample].mean(axis=0)
_, _, axes = np.linalg.svd(centred, full_matrices=False)
xy = centred @ axes[:2].T
with open("code/20/08_word_vectors.json", "w") as f:
    json.dump({"scores": scores, "x": xy[:, 0].round(3).tolist(),
               "y": xy[:, 1].round(3).tolist(),
               "category": train.category.iloc[sample].tolist()}, f)
