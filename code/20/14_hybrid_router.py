# The hybrid: the trained model first, the unsure tickets sent on.
import json
from pathlib import Path

import pandas as pd

from foresight.triage.evaluate import eval_set
from foresight.triage.features import (pemberton, relabel, split,
                                       tickets)
from foresight.triage.model import TriageModel
from foresight.triage.router import THRESHOLD, unsure

t = tickets()
t["category"] = relabel(t)
train, valid, test = split(t)


def right(p, rows):
    return ((p.priority.to_numpy() == rows.priority.to_numpy())
            & (p.category.to_numpy() == rows.category.to_numpy()))


# 1. Choose the threshold on validation, the model trained before it.
p = TriageModel().fit(train.body, train.priority,
                      train.category).predict(valid.body)
hit, cloth = right(p, valid), pemberton(valid).to_numpy()
print("validation: share sent on, and both labels right among the")
print("tickets kept and the tickets sent")
print(f"{'threshold':>11}{'ordinary':>10}{'Pemberton':>11}{'kept':>8}"
      f"{'sent':>8}")
chosen = {}
for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
    send = unsure(p, th).to_numpy()
    if th == THRESHOLD:
        chosen = {"ordinary": send[~cloth].mean(),
                  "pemberton": send[cloth].mean()}
    print(f"{th:11.1f}{send[~cloth].mean():10.1%}"
          f"{send[cloth].mean():11.1%}{hit[~send].mean():8.1%}"
          f"{hit[send].mean():8.1%}")

# 2. The chosen threshold on the comparison set, trained on all before.
TH = THRESHOLD                  # 0.7, recorded in router.py
seen = pd.concat([train, valid], ignore_index=True)
sample = eval_set(test)
p = TriageModel().fit(seen.body, seen.priority,
                      seen.category).predict(sample.body)
hit, send = right(p, sample), unsure(p, TH).to_numpy()
r, kept, alone = send.mean(), hit[~send].sum(), hit.mean()
print(f"\ncomparison set at {TH}: {send.sum()} of {len(sample):,} sent "
      f"on ({r:.1%});")
print(f"the model gets {hit[~send].mean():.1%} of the rest and "
      f"{hit[send].mean():.1%} of those right")
print("\nhybrid accuracy if the LLM gets a share of the sent tickets")
print("right (assumed shares, not measured)")
for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
    total = (kept + a * send.sum()) / len(sample)
    print(f"  {a:4.0%} of the sent -> {total:.1%}   (model alone "
          f"{alone:.1%})")
llm = Path("code/20/11_llm_classifier.json")
if llm.exists():
    q = pd.DataFrame(json.loads(llm.read_text())["eval"])
    both = hit.copy()
    both[send] = right(q, sample)[send]
    print(f"  measured: {both.mean():.1%}")

# 3. What each costs a month. Every figure here is an assumption.
per_month = len(test) / 12
upkeep = 4 * 100            # engineer hours a month x dollars an hour
print(f"\n2025 brought {per_month:.0f} tickets a month. Assumed: the "
      f"trained model")
print(f"costs ${upkeep} a month to keep, and the LLM the first column "
      f"per")
print("1,000 tickets. Meridian's monthly LLM bill, and the monthly")
print("volume at which each design costs the same as the LLM alone:")
print(f"{'$ per 1,000':>13}{'Meridian':>11}"
      f"{'trained':>12}{'hybrid':>12}")
for c in [0.5, 2, 10, 50]:
    llm_month = per_month * c / 1000
    print(f"{c:13.2f}{llm_month:11.2f}"
          f"{upkeep / (c / 1000):12,.0f}"
          f"{upkeep / ((1 - r) * c / 1000):12,.0f}")
with open("code/20/14_hybrid_router.json", "w") as f:
    json.dump({"threshold": TH, "sent": r,
               "kept_right": hit[~send].mean(),
               "sent_right": hit[send].mean(), "validation": chosen},
              f, indent=1)
