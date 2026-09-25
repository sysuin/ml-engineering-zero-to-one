# Exercise 3: a seventh category, Leaving, and the examples it needs.
import re

import pandas as pd

from foresight.config import SEED
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel

LEAVING = re.compile(r"not be renewing|another supplier|close our "
                     r"account|cancel auto renewal", re.I)
t = tickets()
t["category"] = relabel(t)
t.loc[t.body.str.contains(LEAVING), "category"] = "Leaving"
train, valid, _ = split(t)
print(f"Leaving: {(train.category == 'Leaving').sum()} training "
      f"tickets, {(valid.category == 'Leaving').sum()} validation")

leave = train[train.category == "Leaving"]
rest = train[train.category != "Leaving"]
print(f"{'examples':>9}{'recall':>9}{'precision':>11}")
for n in [2, 5, 10, 20, 50, len(leave)]:
    part = pd.concat([rest, leave.sample(n, random_state=SEED)])
    p = TriageModel().fit(part.body, part.priority,
                          part.category).predict(valid.body)
    said = p.category.to_numpy() == "Leaving"
    real = (valid.category == "Leaving").to_numpy()
    print(f"{n:9}{(said & real).sum() / real.sum():9.1%}"
          f"{(said & real).sum() / max(said.sum(), 1):11.1%}")
