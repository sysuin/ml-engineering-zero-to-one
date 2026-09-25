# The milestone's one look at the test year: v0.5's model, the lasso
# on Chapter 4's columns, with the rule, Chapter 12's lasso+ and the
# booster on the same rows; then the Pemberton column, by quarter.
# timeout: 300
import json
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE, ROOT, TRUTH
from foresight.data.build_table import legacy_ids
from foresight.evaluate import hits_at_k, report
from foresight.features.build import load
from foresight.features.registry import names
from foresight.features.sources import Context, Sources
from foresight.leaderboard import paired, point, run
from foresight.models.featured import booster, lasso

# The Pemberton column, exactly as 06_pemberton_feature builds it.
con = sqlite3.connect(ML_WAREHOUSE)
t = pd.read_sql_query("""SELECT account_id, date(opened_at) AS day,
    category, body FROM tickets""", con)
t["account_id"] = t.account_id.replace(legacy_ids(con))
t["day"] = pd.to_datetime(t.day)
cloth = t[t.category.isin(["Quality", "Delivery"])
          & t.body.str.contains("MRD-CLE-001", case=False)]
table = load()
none = pd.DataFrame(columns=["account_id", "day"])
ctx = Context(table, Sources(none, none, cloth, none))
e = ctx.window("tickets", 90)
table["pemberton_90d"] = ctx.per_contract(
    e.groupby("contract_id").size()).to_numpy()

# Chosen before this run: lasso+ as Chapter 12's leaderboard chose it.
kept = names(("windows", "trends", "interactions", "behaviour"))
board = run({"v0.5": lasso(), "lasso+": lasso(kept, 0.005),
             "booster": booster(),
             "with pemberton": lasso(["pemberton_90d"])},
            split="test", table=table)
PAGE = ROOT / "docs" / "foresight-v0.5-test.md"
print(report(board["runs"]["v0.5"], PAGE,
             "Foresight v0.5 on the test year: reported once"))

lines = ["", f"{'Test year, 480 calls':<22}{'leavers':>8}{'AUC':>7}"
         "  minus v0.5, paired"]
hits = {}
for name in board["runs"]:
    hits[name] = round(point(board, "precision", name) * 480)
    a = point(board, "auc", name)
    cell = ""
    if name != "v0.5":
        d, lo, hi = paired(board, "precision", name, "v0.5")
        cell = f"  {d * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f})"
    lines.append(f"  {name:<20}{hits[name]:>8}{a:>7.3f}{cell}")
for name in ("lasso+", "booster", "with pemberton"):
    d, lo, hi = paired(board, "auc", name, "v0.5")
    lines.append(f"  AUC, {name} - v0.5: {d:+.3f}"
                 f" ({lo:+.3f} to {hi:+.3f})")
scored = board["runs"]["v0.5"]["scored"]
truth = pd.read_csv(TRUTH / "renewals.csv").set_index("contract_id")
best = scored.assign(model=truth.p_leave.reindex(
    scored.contract_id).to_numpy())
ceiling = sum(hits_at_k(c.not_renewed, c.model, c.contract_id)
              for _, c in best.groupby("moment"))
lines += [f"  ceiling (true chances){ceiling:>6}", "",
          "The test year is now spent for v0.5."]
print("\n".join(lines))
with open(PAGE, "a") as f:
    f.write("\n```text" + "\n".join(lines) + "\n```\n")

# The Pemberton column by quarter of mark, 2023 to 2025.
q = table.assign(quarter=table.moment.dt.to_period("Q"),
                 flag=table.pemberton_90d > 0)
print("\nLeft, by quarter of the mark    with a cloth"
      " complaint    without")
out = []
for quarter, c in q.groupby("quarter"):
    w, wo = c[c.flag], c[~c.flag]
    rate = f"{w.not_renewed.mean():>8.1%} of {len(w):<4}" if len(w) \
        else f"{'-':>8}{'':8}"
    print(f"  {str(quarter):<30}{rate}{wo.not_renewed.mean():>9.1%}")
    out.append([str(quarter), len(w), float(w.not_renewed.sum()),
                len(wo), float(wo.not_renewed.sum())])
with open("code/13/12_test_report.json", "w") as f:
    json.dump({"quarters": out, "ceiling": ceiling, "hits": hits}, f)
