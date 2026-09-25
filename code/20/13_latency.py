# How long the trained models take: to train, per ticket, per thousand.
# nondeterministic: wall-clock timing
import json
import time

import numpy as np
import pandas as pd

from foresight.config import seed_everything
from foresight.triage.evaluate import eval_set
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel
from foresight.triage.neural import TransformerTriage

seed_everything()
t = tickets()
t["category"] = relabel(t)
train, valid, test = split(t)
seen = pd.concat([train, valid], ignore_index=True)
bodies = eval_set(test).body.tolist()

print(f"{'':13}{'training':>10}{'one ticket':>12}{'1,000 at once':>15}")
timings = {}
for name, model in [("TF-IDF", TriageModel()),
                    ("transformer", TransformerTriage())]:
    started = time.process_time()
    model.fit(seen.body, seen.priority, seen.category)
    train_s = time.process_time() - started
    one = []
    for body in bodies[:200]:           # as tickets arrive, one by one
        started = time.perf_counter()
        model.predict([body])
        one.append(time.perf_counter() - started)
    started = time.process_time()
    model.predict(bodies)               # a day's backlog, in one call
    batch_s = time.process_time() - started
    timings[name] = {"train_s": train_s, "one_ms": 1e3 * np.median(one),
                     "per_1000_s": batch_s}
    print(f"  {name:11}{train_s:9.1f}s{1e3 * np.median(one):10.2f}ms"
          f"{batch_s:14.3f}s")
with open("code/20/13_latency.json", "w") as f:
    json.dump(timings, f, indent=1)
