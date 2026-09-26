# The router on the half-year that had the unusual month: every ticket
# it sends on from July to December 2024, answered by the language
# model.
# nondeterministic: live model calls
# timeout: 3600
import json

from foresight.triage.evaluate import paired
from foresight.triage.features import (PRIORITIES, pemberton, relabel,
                                       split, tickets)
from foresight.triage.llm import Budget, classify
from foresight.triage.model import TriageModel
from foresight.triage.router import THRESHOLD, unsure

t = tickets()
t["category"] = relabel(t)
train, valid, _ = split(t)
p = TriageModel().fit(train.body, train.priority,
                      train.category).predict(valid.body)
send = unsure(p, THRESHOLD).to_numpy()
cloth = pemberton(valid).to_numpy()
sent = valid[send].reset_index(drop=True)

budget = Budget(ceiling=1.00)   # about 45 cents; a dollar at most
answers = classify(sent.body, budget=budget)
ok = (answers.category.notna() & answers.priority.notna()).to_numpy()


def right(q, rows, label=None):
    labels = [label] if label else ["priority", "category"]
    hit = True
    for col in labels:
        hit = hit & (q[col].to_numpy() == rows[col].to_numpy())
    return hit


model = right(p, valid)                 # the trained model alone
hybrid = model.copy()                   # the LLM where it answered
hybrid[send] = (right(answers, sent) & ok) | (model[send] & ~ok)

print(f"July to December 2024, threshold {THRESHOLD}: "
      f"{send.sum():,} of {len(valid):,} sent on")
print(f"{ok.mean():.1%} of replies fit the schema\n")
print("both labels right on the sent tickets")
print(f"{'':12}{'sent':>7}{'model':>8}{'LLM':>8}{'LLM - model':>15}")
for name, part in [("Pemberton", cloth[send]),
                   ("ordinary", ~cloth[send])]:
    m, h = model[send][part], hybrid[send][part]
    d, lo, hi = paired(h, m)
    print(f"  {name:10}{part.sum():7,}{m.mean():8.1%}{h.mean():8.1%}"
          f"{100 * d:+7.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})")

print("\none label at a time, Pemberton tickets sent")
rows, q, pm = sent[cloth[send]], answers[cloth[send]], p[send & cloth]
for label in ["category", "priority"]:
    print(f"  {label:9} model {right(pm, rows, label).mean():.1%}, "
          f"LLM {right(q, rows, label).mean():.1%}")
rank = {k: i for i, k in enumerate(PRIORITIES)}
said = q.priority.map(rank).to_numpy()
desk = rows.priority.map(rank).to_numpy()
print(f"  LLM priority above the desk's {(said < desk).sum()}, "
      f"below {(said > desk).sum()}")

autumn = (valid.opened_at >= "2024-10-01").to_numpy()
print("\nboth labels right on every ticket, the model alone and routed")
print(f"{'':24}{'tickets':>7}{'model':>8}{'hybrid':>8}")
for name, part in [("all Pemberton tickets", cloth),
                   ("October to December", autumn),
                   ("July to December", autumn | ~autumn)]:
    print(f"  {name:22}{part.sum():7,}{model[part].mean():8.1%}"
          f"{hybrid[part].mean():8.1%}")
print(f"\nspent ${budget.spent:.4f} of the $1.00 ceiling")

with open("code/20/14b_router_on_pemberton.json", "w") as f:
    json.dump({"ticket_id": sent.ticket_id.tolist(),
               "pemberton": cloth[send].tolist(),
               "answers": answers.to_dict(orient="records"),
               "spent_usd": budget.spent}, f)
