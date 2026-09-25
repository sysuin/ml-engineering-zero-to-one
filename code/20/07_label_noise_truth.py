# How noisy the desks' labels are, against the generator's truth.
# The only listing in the chapter that reads truth/tickets.csv. Nothing
# here is used to train or to choose; it measures the labels themselves,
# and the ceiling they put on any model trained and scored on them.
import pandas as pd

from foresight.config import TRUTH
from foresight.triage.evaluate import urgent_at_capacity
from foresight.triage.features import relabel, split, tickets
from foresight.triage.model import TriageModel

t = tickets()
t["rubric"] = relabel(t)
truth = pd.read_csv(TRUTH / "tickets.csv")
t = t.merge(truth, on="ticket_id")
t["true_priority"] = t.true_urgency

print("desk labels that match the truth, every ticket")
rows = {"category as filed": (t.category == t.true_category),
        "category, rubric": (t.rubric == t.true_category),
        "priority": (t.priority == t.true_priority)}
for name, hit in rows.items():
    d = hit.groupby(t.desk).mean()
    print(f"  {name:18}{hit.mean():6.1%}   North {d.iloc[0]:.1%}"
          f"   South {d.iloc[1]:.1%}")

filed = t[t.priority == "Urgent"]
real = t[t.true_priority == "Urgent"]
print(f"\nfiled Urgent {len(filed):,}: truly Urgent "
      f"{(filed.true_priority == 'Urgent').mean():.1%}")
print(f"truly Urgent {len(real):,}: filed Urgent "
      f"{(real.priority == 'Urgent').mean():.1%}")

train, valid, _ = split(t)
model = TriageModel().fit(train.body, train.priority, train.rubric)
pred = model.predict(valid.body)
print("\nthe baseline on validation (trained on desk labels, rubric")
print("categories), scored two ways")
print(f"{'':22}{'desk labels':>12}{'truth':>9}")
for label, desk, true in [("priority", "priority", "true_priority"),
                          ("category", "rubric", "true_category")]:
    print(f"  {label + ' accuracy':20}"
          f"{(pred[label] == valid[desk]).mean():12.1%}"
          f"{(pred[label] == valid[true]).mean():9.1%}")
desk = urgent_at_capacity(valid, pred.p_urgent)
true = urgent_at_capacity(valid, pred.p_urgent, "true_priority")
print(f"  {'Urgent, 3 a day':20}{desk:12.1%}{true:9.1%}")
