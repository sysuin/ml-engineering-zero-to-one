# Trained models on the comparison set: accuracy and unfamiliar input.
import json
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import rng, seed_everything
from foresight.triage.evaluate import eval_set
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel
from foresight.triage.neural import TransformerTriage
from foresight.triage.probes import probes

seed_everything()
t = tickets()
t["category"] = relabel(t)
train, valid, test = split(t)
seen = pd.concat([train, valid], ignore_index=True)   # all before 2025
sample, probe = eval_set(test), probes()

answers = {}
for name, model in [("TF-IDF", TriageModel()),
                    ("transformer", TransformerTriage())]:
    model.fit(seen.body, seen.priority, seen.category)
    answers[name] = (model.predict(sample.body),
                     model.predict(probe.body))
llm = Path("code/20/11_llm_classifier.json")
if llm.exists():
    run = json.loads(llm.read_text())
    answers["LLM"] = (pd.DataFrame(run["eval"]),
                      pd.DataFrame(run["probes"]))


def interval(hit):
    g = rng()
    draws = [hit[g.integers(0, len(hit), len(hit))].mean()
             for _ in range(2000)]
    return np.quantile(draws, [0.025, 0.975])


print(f"{len(sample):,} tickets from 2025, desk priorities and rubric "
      f"categories")
print(f"{'':20}" + "".join(f"{n:>16}" for n in answers))
result = {n: {} for n in answers}
for label in ["priority", "category", "both"]:
    cells, spans = "", ""
    for n, (p, _) in answers.items():
        if label == "both":
            hit = ((p.priority == sample.priority)
                   & (p.category == sample.category)).to_numpy()
        else:
            hit = (p[label] == sample[label]).to_numpy()
        lo, hi = interval(hit)
        result[n][label] = [hit.mean(), lo, hi]
        cells += f"{hit.mean():>16.1%}"
        span = f"{100 * lo:.1f} to {100 * hi:.1f}"
        spans += f"{span:>16}"
    print(f"  {label + ' accuracy':18}{cells}")
    print(f"  {'  95% interval':18}{spans}")
urgent = (sample.priority == "Urgent").to_numpy()
for what in ["recall", "precision"]:
    cells = ""
    for n, (p, _) in answers.items():
        said = (p.priority == "Urgent").to_numpy()
        base = urgent if what == "recall" else said
        v = (said & urgent).sum() / max(base.sum(), 1)
        result[n][f"urgent_{what}"] = v
        cells += f"{v:>16.1%}"
    print(f"  {'Urgent ' + what:18}{cells}")

print(f"\n{len(probe)} probes (both labels right, by kind)")
for kind in probe.kind.unique():
    k = (probe.kind == kind).to_numpy()
    cells = ""
    for n, (_, q) in answers.items():
        hit = ((q.priority == probe.priority)
               & (q.category == probe.category)).to_numpy()
        result[n][kind] = int(hit[k].sum())
        cells += f"{hit[k].sum():>11} of {k.sum()}"
    print(f"  {kind:18}{cells}")
print("\nmean confidence in the category")
for where, j in [("comparison set", 0), ("probes", 1)]:
    cells = ""
    for n, pair in answers.items():
        conf = pair[j].get("category_conf")
        cells += (f"{conf.mean():>16.2f}" if conf is not None
                  else f"{'-':>16}")
    print(f"  {where:18}{cells}")
if "LLM" not in answers:
    print("\nLLM: awaiting a run (11_llm_classifier needs an API key)")
with open("code/20/12_four_axes.json", "w") as f:
    json.dump(result, f, indent=1)
