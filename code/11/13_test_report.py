# The milestone's one look at the test year: v0.4's model, the lasso,
# on 2025's cohorts, paired with v0.3 and the rule on the same rows.
# timeout: 300
import pandas as pd

from foresight.config import ROOT, TRUTH
from foresight.evaluate import hits_at_k, report
from foresight.leaderboard import paired, run
from foresight.models.logistic import RenewalRisk
from foresight.models.regularised import maker

PAGE = ROOT / "docs" / "foresight-v0.4-test.md"
board = run({"lasso": maker("l1", 0.002), "v0.3": RenewalRisk},
            split="test")
text = report(board["runs"]["lasso"], PAGE,
              "Foresight v0.4 on the test year: reported once")
print(text)

lines = ["", "v0.4 (the lasso) minus v0.3 (logistic), paired"]
for what, name in (("precision", "precision at 40"), ("auc", "AUC")):
    d, lo, hi = paired(board, what, "lasso", "v0.3")
    if what == "precision":
        lines.append(f"  {name:<18}{d * 100:+.1f} points"
                     f" ({lo * 100:+.1f} to {hi * 100:+.1f})")
    else:
        lines.append(f"  {name:<18}{d:+.3f} ({lo:+.3f} to {hi:+.3f})")

scored = board["runs"]["lasso"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
scored = scored.assign(best=truth.p_leave.reindex(
    scored.contract_id).to_numpy())
ceiling = sum(hits_at_k(c.not_renewed, c.best, c.contract_id)
              for _, c in scored.groupby("moment"))
hits = {n: sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
               for _, c in board["runs"][n]["scored"].groupby("moment"))
        for n in board["runs"]}
lines += ["", f"Ceiling: {ceiling} leavers in the top 40s, from the"
          " generator's", f"true probabilities. v0.4 reached "
          f"{hits['lasso']}, v0.3 {hits['v0.3']}.",
          "", "The test year is now spent for v0.4."]
print("\n".join(lines))
with open(PAGE, "a") as f:
    f.write("\n```text" + "\n".join(lines) + "\n```\n")
