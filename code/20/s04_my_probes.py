# Exercise 4: ten more probes, labelled before running, and the router.
import pandas as pd

from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel
from foresight.triage.router import unsure

MINE = [  # body, category, priority, written before any model saw them
    ("the pallet jack snapped a wheel and the load fell", "Quality",
     "Urgent"),
    ("half the soap dispensers leak, and invoice INV-20411 is wrong "
     "too", "Quality", "High"),
    ("can you stop sending paper statements", "Account", "Low"),
    ("rain got into the van, all our towels are soggy", "Delivery",
     "Normal"),
    ("we're closing the kitchen at noon unless the gloves come",
     "Stock", "Urgent"),
    ("the courier says order 1400123 was left at reception, it wasn't",
     "Delivery", "Normal"),
    ("Hemos pagado dos veces la factura INV-30112", "Billing",
     "Normal"),
    ("Tilaus 1400456 ei ole vielä saapunut", "Delivery", "Normal"),
    ("do you do a recycled version of MRD-PAC-021?", "Stock", "Normal"),
    ("please collect the wrong mops, booking ref 5521", "Returns",
     "Normal"),
]
t = tickets()
t["category"] = relabel(t)
train, valid, _ = split(t)
seen = pd.concat([train, valid])
mine = pd.DataFrame(MINE, columns=["body", "category", "priority"])
p = TriageModel().fit(seen.body, seen.priority,
                      seen.category).predict(mine.body)
right = ((p.category == mine.category)
         & (p.priority == mine.priority)).to_numpy()
sent = unsure(p).to_numpy()
for i, r in mine.iterrows():
    mark = "sent" if sent[i] else "kept"
    print(f"  {mark} {'ok ' if right[i] else 'bad'} "
          f"{p.category[i][:8]:8} {p.priority[i]:7} "
          f"{p.category_conf[i]:.2f}  {r.body[:30]}")
print(f"both right {right.sum()} of {len(mine)}; sent {sent.sum()}; "
      f"kept and wrong {(~sent & ~right).sum()}")
