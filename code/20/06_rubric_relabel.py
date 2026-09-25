# The rubric applied: which labels move, and what agreement it buys.
import pandas as pd

from foresight.triage.evaluate import desk_pairs, kappa
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel

t = tickets()
t["rubric"] = relabel(t)
moved = t[t.rubric != t.category]
print(f"the rubric moves {len(moved):,} of {len(t):,} labels")
change = (moved.category + " -> " + moved.rubric).rename("label")
print(pd.crosstab(change, moved.desk).to_string())

seen = t[t.opened_at < "2025-01-01"]
print("\nnear-identical pairs across the desks, before 2025")
for name, column in [("as filed", "category"), ("rubric", "rubric")]:
    pairs = desk_pairs(seen, label=column)
    same = (pairs.north == pairs.south).mean()
    left = ((pairs.north == "Quality")
            & (pairs.south == "Delivery")).sum()
    print(f"  {name:9} same label {same:.1%}  kappa "
          f"{kappa(pairs.north, pairs.south):.3f}  "
          f"Quality/Delivery {left}")

train, valid, _ = split(t)
print("\ncategory accuracy on validation, each model scored on the")
print("labels it was trained to reproduce")
for name, column in [("as filed", "category"), ("rubric", "rubric")]:
    model = TriageModel().fit(train.body, train.priority, train[column])
    hit = model.predict(valid.body).category.to_numpy() == valid[column]
    by_desk = hit.groupby(valid.desk).mean()
    print(f"  {name:9} {hit.mean():.1%}   North {by_desk.iloc[0]:.1%}"
          f"   South {by_desk.iloc[1]:.1%}")
