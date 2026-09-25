# Tickets not in English: what each trained model makes of them.
from foresight.config import seed_everything
from foresight.triage.features import relabel, split, tickets, tokens
from foresight.triage.model import TriageModel
from foresight.triage.neural import TransformerTriage
from foresight.triage.probes import probes

seed_everything()
t = tickets()
t["category"] = relabel(t)
train, valid, _ = split(t)
foreign = train[train.language != "en"]
masked = foreign.body.map(lambda b: " ".join(tokens(b))).nunique()
print(f"training: {len(foreign)} of {len(train):,} tickets not in "
      f"English; with codes and")
print(f"numbers masked, they are {masked} different sentences")

models = {"TF-IDF": TriageModel(), "transformer": TransformerTriage()}
for m in models.values():
    m.fit(train.body, train.priority, train.category)

print("\nvalidation, both labels right")
print(f"{'':10}{'tickets':>8}" + "".join(f"{n:>13}" for n in models))
preds = {n: m.predict(valid.body) for n, m in models.items()}
for lang, rows in valid.groupby("language"):
    cells = ""
    for p in preds.values():
        ok = ((p.priority.to_numpy()[rows.index] == rows.priority)
              & (p.category.to_numpy()[rows.index] == rows.category))
        cells += f"{ok.mean():13.1%}"
    print(f"  {lang:8}{len(rows):8}{cells}")

pr = probes()
pr = pr[pr.kind != "new wording"].reset_index(drop=True)
print("\nnew sentences, written for this test: category given, and the")
print("model's confidence in it")
out = {n: m.predict(pr.body) for n, m in models.items()}
print(f"{'':24}{'rubric':>9}" + "".join(f"{n:>15}" for n in models))
for i, r in pr.iterrows():
    if i == 0 or r.kind != pr.kind[i - 1]:
        print(f"  {r.kind}")
    cells = "".join(f"  {out[n].category[i][:8]:>8} "
                    f"{out[n].category_conf[i]:.2f}" for n in models)
    print(f"  {r.body[:22]:22}{r.category[:8]:>9}{cells}")
for n in models:
    right = (out[n].category == pr.category).to_numpy()
    new = (pr.kind == "new language").to_numpy()
    print(f"  {n}: {right[~new].sum()} of {(~new).sum()} in a known "
          f"language, {right[new].sum()} of {new.sum()} in a new one")
