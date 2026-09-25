# Every group of features, judged on the training period only, and
# the rule that decides which groups each model keeps.
# timeout: 300
from foresight.features.build import load
from foresight.features.registry import groups, names
from foresight.models.featured import NOISE, TRAINING, evidence, lasso

table = load()
candidates = {g: names(g) for g in groups()}
candidates["everything"] = names()
ev = evidence(table, candidates)

print(f"Change on the training period{'':4}{'lasso, tuning':>16}"
      f"{'booster':>10}")
print(f"{'group':<15}{'features':>8}{'log loss':>11}{'AUC':>8}"
      f"{'watched':>10}  keep")
for g, r in ev.iterrows():
    keep = ("lasso " if r["keep lasso"] else "") + \
        ("booster" if r["keep booster"] else "")
    print(f"  {g:<13}{r.features:>8}{r.lasso:>+11.5f}"
          f"{r['lasso auc']:>+8.3f}{r.booster:>+10.5f}  {keep}"
          .rstrip())
print(f"\nThe rule: keep a group for a model if it lowers that model's"
      f"\nlog loss on the training period by more than {NOISE}.")

# The lasso's own opinion of the whole library: given every feature,
# how many does it keep when fitted on the training split?
w = lasso(names())().fit(
    table[table.end_date.between(*TRAINING)]).weights()[names()]
print(f"\nThe lasso, given all {len(w)} library features, keeps"
      f" {(w != 0).sum()};\n  largest weights:"
      f" {', '.join(w.abs().sort_values().index[-3:][::-1])}")
