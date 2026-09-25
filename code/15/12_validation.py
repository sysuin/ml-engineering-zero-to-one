# The tuned booster read once on the validation cohorts, beside v0.4's
# lasso and Chapter 11's booster, by Chapter 8's backtest with paired
# intervals, then Chapter 11's rule and the ceiling.
# timeout: 300
import pandas as pd

from foresight.config import TRUTH
from foresight.data.build_table import TABLE
from foresight.evaluate import auc, hits_at_k
from foresight.leaderboard import paired, point, run
from foresight.models.boosting import RenewalBooster
from foresight.models.regularised import maker
from foresight.tune import best, booster, search

table = pd.read_parquet(TABLE)
tuned = best(search(table))             # chosen on training cohorts
board = run({"lasso": maker("l1", 0.002), "booster": RenewalBooster,
             "tuned": booster(tuned)}, table=table)

print(f"{'':10}{'leavers':>8}  {'precision at 40':<18}AUC")
for name in ("lasso", "booster", "tuned", "rule"):
    hits = round(point(board, "precision", name) * 240)
    print(f"{name:<10}{hits:>8}  "
          f"{point(board, 'precision', name) * 100:<18.1f}"
          f"{point(board, 'auc', name):.3f}")

print(f"\n{'Paired, 95%':<18}{'precision, points':<22}AUC")
for a, b in (("tuned", "booster"), ("tuned", "lasso")):
    d, lo, hi = paired(board, "precision", a, b)
    x, xlo, xhi = paired(board, "auc", a, b)
    print(f"{a + ' - ' + b:<18}"
          f"{d * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f})"
          f"{'':4}{x:+.3f} ({xlo:+.3f} to {xhi:+.3f})")

# Chapter 11's rule: replace the lasso only if not behind it at
# capacity and ahead on one paired interval clear of zero.
behind = point(board, "precision", "tuned") < point(
    board, "precision", "lasso")
ahead = any(paired(board, w, "tuned", "lasso")[1] > 0
            for w in ("precision", "auc"))
verdict = "replaces" if ahead and not behind else "does not replace"
print(f"\nThe tuned booster {verdict} the lasso")

scored = board["runs"]["tuned"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
p = truth.p_leave.reindex(scored.contract_id).to_numpy()
ceiling = sum(hits_at_k(c.not_renewed, p[i], c.contract_id)
              for i, (_, c) in zip(
                  scored.groupby("moment").indices.values(),
                  scored.groupby("moment")))
print(f"Ceiling: {ceiling} leavers, AUC"
      f" {auc(scored.not_renewed, p):.3f}")
