# A small transformer, trained from scratch on a laptop CPU.
import json

import numpy as np

from foresight.config import seed_everything
from foresight.triage.evaluate import paired
from foresight.triage.features import (pemberton, relabel, split,
                                       tickets)
from foresight.triage.model import TriageModel
from foresight.triage.neural import TransformerTriage

seed_everything()
t = tickets()
t["category"] = relabel(t)
train, valid, _ = split(t)
watch = (valid.body, valid.priority, valid.category)
net = TransformerTriage().fit(
    train.body, train.priority, train.category, watch=watch)
size = sum(p.numel() for p in net.net_.parameters())
print(f"{size:,} parameters; accuracy on validation after each epoch")
print(f"{'epoch':>7}{'train loss':>12}{'priority':>10}{'category':>10}")
for e, (loss, (p, c)) in enumerate(zip(net.losses_, net.watched_), 1):
    print(f"{e:7}{loss:12.3f}{p:10.1%}{c:10.1%}")

tfidf = TriageModel().fit(train.body, train.priority, train.category)
cloth = pemberton(valid).to_numpy()
print(f"\n{'':15}{'priority':>18}{'category':>20}")
print(f"{'':15}" + f"{'ordinary':>11}{'cloth':>8}" * 2)
hits, scores = {}, {}
for name, model in [("TF-IDF", tfidf), ("transformer", net)]:
    p, cells = model.predict(valid.body), ""
    for label in ["priority", "category"]:
        hit = np.asarray(p[label]) == valid[label].to_numpy()
        hits[name, label] = hit
        scores[f"{name} {label}"] = [hit[~cloth].mean(),
                                     hit[cloth].mean()]
        cells += f"{hit[~cloth].mean():11.1%}{hit[cloth].mean():8.1%}"
    print(f"  {name:13}{cells}")

print("\ntransformer minus TF-IDF, ordinary tickets, 95% interval")
for label in ["priority", "category"]:
    d, lo, hi = paired(hits["transformer", label][~cloth],
                       hits["TF-IDF", label][~cloth])
    print(f"  {label:9}{100 * d:+.1f} points  ({100 * lo:+.1f} to "
          f"{100 * hi:+.1f})")
with open("code/20/09_small_transformer.json", "w") as f:
    json.dump({"parameters": size, "scores": scores}, f, indent=1)
