# Meridian's manifest, its tables, and the best any model can do.
import json
import sqlite3

import pandas as pd

from foresight.config import MERIDIAN_ML, ML_WAREHOUSE, TRUTH
from foresight.evaluate import auc, hits_at_k

manifest = json.loads((MERIDIAN_ML / "manifest.json").read_text())
print(f"seed {manifest['seed']}, synthetic: {manifest['synthetic']},"
      f" personal data: {manifest['contains_personal_data']}")
for name, n in manifest["counts"].items():
    print(f"  {name:<24}{n:>10,}")

con = sqlite3.connect(ML_WAREHOUSE)
print("\nRows in the warehouse")
for table in ("accounts", "account_history",
              "products", "suppliers", "regions", "price_events"):
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<24}{n:>10,}")
running = con.execute("SELECT COUNT(*) FROM contracts"
                      " WHERE outcome IS NULL").fetchone()[0]
print(f"  contracts still running{running:>11,}")

r = manifest["renewals"]
print(f"\nLong-tail renewals decided, 2023-2025"
      f"{r['decided_2023_2025_long_tail']:>10,}")
print(f"  not renewed{r['not_renewed_rate']:>34.1%}")
print(f"  ceiling AUC{r['ceiling_auc']:>34.3f}")

# The truth file, read here only to measure the ceiling.
truth = pd.read_csv(TRUTH / "renewals.csv")
truth = truth[truth.outcome.notna()]
truth["left"] = (truth.outcome == "not_renewed").astype(int)
tail = truth[(truth.key_account == 0)
             & (truth.end_date >= "2023-01-01")]
by_year = tail.groupby(tail.end_date.str[:4]).left.mean()
print("  by year ending  " + "  ".join(
    f"{y} {v:.1%}" for y, v in by_year.items()))

print("\nThe ceiling at 40 calls a cohort, every renewal")
print(f"{'split':<11}{'contracts':>10}{'leavers':>9}"
      f"{'ceiling':>10}{'precision':>11}{'AUC':>7}")
for split, lo, hi in (("train", "2023-01-01", "2024-06-30"),
                      ("validation", "2024-07-01", "2024-12-31"),
                      ("test", "2025-01-01", "2025-12-31")):
    s = truth[(truth.end_date >= lo) & (truth.end_date <= hi)]
    cohorts = s.groupby(s.end_date.str[:7])
    hits = sum(hits_at_k(c.left, c.p_leave, c.contract_id)
               for _, c in cohorts)
    calls = 40 * cohorts.ngroups
    print(f"{split:<11}{len(s):>10,}{s.left.sum():>9}"
          f"{f'{hits}/{calls}':>10}{hits / calls:>11.1%}"
          f"{auc(s.left, s.p_leave):>7.3f}")
