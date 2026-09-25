# The baseline's category mistakes, by the desk that filed each ticket.
from foresight.triage.features import split, tickets
from foresight.triage.model import TriageModel

train, valid, _ = split(tickets())
model = TriageModel().fit(train.body, train.priority, train.category)
valid["predicted"] = model.predict(valid.body).category.to_numpy()

print("category accuracy on validation, by desk")
for desk, rows in valid.groupby("desk"):
    print(f"  {desk:11}{(rows.predicted == rows.category).mean():.1%}"
          f"  of {len(rows):,}")

wrong = valid[valid.predicted != valid.category]
cells = (wrong.groupby(["desk", "category", "predicted"]).size()
         .rename("tickets").reset_index()
         .sort_values("tickets", ascending=False))
print("\nthe commonest mistakes (desk's label -> model's)")
for _, r in cells.head(6).iterrows():
    print(f"  {r.desk:11}{r.category:>9} -> {r.predicted:9}"
          f"{r.tickets:5}")

pair = wrong[wrong.category.isin(["Quality", "Delivery"])
             & wrong.predicted.isin(["Quality", "Delivery"])]
print(f"\n{len(pair)} of {len(wrong)} mistakes are Quality/Delivery; "
      f"a few of them:")
for _, r in pair.groupby("desk").head(2).iterrows():
    print(f"  {r.desk[:5]} {r.category[:4]}  {r.body[:52]}")
