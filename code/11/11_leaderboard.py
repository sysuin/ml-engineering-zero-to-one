# v0.4's leaderboard: the rule and four models by Chapter 8's backtest,
# then the ceiling, read from the generator for this comparison only.
# timeout: 300
import json

import pandas as pd

from foresight.config import TRUTH
from foresight.evaluate import auc, hits_at_k, interval
from foresight.leaderboard import (BOARD, draws, paired, point,
                                   report, run)
from foresight.models.boosting import contenders

board = run(contenders())
print(report(board, BOARD))

print(f"\n{'':20}{'precision, points':<22}AUC")
for a, b in (("boosting", "lasso"), ("boosting", "forest")):
    cells = []
    for what in ("precision", "auc"):
        d, lo, hi = paired(board, what, a, b)
        if what == "precision":
            d, lo, hi = d * 100, lo * 100, hi * 100
            cells.append(f"{d:+.1f} ({lo:+.1f} to {hi:+.1f})")
        else:
            cells.append(f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"{a + ' - ' + b:<20}{cells[0]:<22}{cells[1]}")

# The best any model could do: rank by the true chance of leaving.
scored = board["runs"]["boosting"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
best = truth.p_leave.reindex(scored.contract_id).to_numpy()
ceiling = sum(hits_at_k(c.not_renewed, best[rows], c.contract_id)
              for rows, (_, c) in zip(
                  scored.groupby("moment").indices.values(),
                  scored.groupby("moment")))
print(f"\nCeiling: {ceiling} leavers in 240 calls,"
      f" AUC {auc(scored.not_renewed, best):.3f}")
for name in board["runs"]:
    hits = round(point(board, "precision", name) * 240)
    print(f"  {name:<10}{ceiling - hits:>3} leavers short of it")

names = [*board["runs"], "rule"]
with open("code/11/11_leaderboard.json", "w") as f:
    json.dump({"ceiling": ceiling, "calls": 240,
               "ceiling_auc": auc(scored.not_renewed, best),
               "random": point(board, "precision", "random"),
               "perfect": point(board, "precision", "perfect"),
               "lists": {n: {w: [point(board, w, n),
                                 *interval(draws(board, w, n))]
                             for w in ("precision", "auc")}
                         for n in names}}, f)
