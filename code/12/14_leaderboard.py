# v0.5's leaderboard: the lasso and the booster with and without the
# groups the rule kept, settings chosen again on the training period.
# timeout: 300
import pandas as pd

from foresight.config import ROOT, TRUTH
from foresight.evaluate import auc, hits_at_k
from foresight.features.build import load
from foresight.features.registry import groups, names
from foresight.leaderboard import paired, point, report, run
from foresight.models.featured import (booster, evidence, lasso,
                                       tune_booster, tune_lasso)

table = load()
ev = evidence(table, {g: names(g) for g in groups()})
kept = {m: [f for g in ev.index[ev[f"keep {m}"]] for f in names(g)]
        for m in ("lasso", "booster")}

# Chapter 9's search and Chapter 11's, run again on the new columns.
tl = tune_lasso(table, kept["lasso"])
strength = float(tl.loc[tl["log loss"].idxmin(), "strength"])
tb = tune_booster(table, kept["booster"])
shape = tb.loc[tb.watched.idxmin(), ["leaves", "min_leaf"]].astype(int)
print(f"lasso+    {len(kept['lasso'])} more columns, strength"
      f" {strength:g}")
print(f"booster+  {len(kept['booster'])} more columns,"
      f" {shape.leaves} leaves of at least {shape.min_leaf}\n")

board = run({"lasso": lasso(), "lasso+": lasso(kept["lasso"],
                                               strength),
             "booster": booster(),
             "booster+": booster(kept["booster"], *shape)},
            table=table)
page = report(board, ROOT / "docs" / "foresight-leaderboard-v0.5.md")
print(page[page.index("           leavers"):page.index("\n\nA diff")])

# The rule from Chapter 11: replace the lasso only if not behind it at
# capacity and ahead on one paired interval clear of zero.
calls = 240
for name in ("lasso+", "booster", "booster+"):
    behind = point(board, "precision", name) < point(
        board, "precision", "lasso")
    ahead = any(paired(board, w, name, "lasso")[1] > 0
                for w in ("precision", "auc"))
    verdict = "replaces" if ahead and not behind else "does not replace"
    print(f"{name:<9} {verdict} the lasso")
d, lo, hi = paired(board, "precision", "booster+", "booster")
a, alo, ahi = paired(board, "auc", "booster+", "booster")
print(f"\nbooster+ - booster {d * 100:+.1f} ({lo * 100:+.1f} to"
      f" {hi * 100:+.1f}) {a:+.3f} ({alo:+.3f} to {ahi:+.3f})")

scored = board["runs"]["lasso"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
best = scored.assign(model=truth.p_leave.reindex(
    scored.contract_id).to_numpy())
ceiling = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
              for _, c in best.groupby("moment"))
print(f"\nCeiling: {ceiling} leavers in {calls} calls,"
      f" AUC {auc(best.not_renewed, best.model):.3f}")
