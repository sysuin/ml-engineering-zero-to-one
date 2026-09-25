# Contracts attached to 2025 sales: a merge that inflates revenue.
import json
import sqlite3
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
sales = pd.read_sql("SELECT order_id, sku, account_id, revenue"
                    " FROM v_sales WHERE order_date >= '2025-01-01'",
                    con)
contracts = pd.read_sql("SELECT contract_id, account_id, start_date,"
                        " term_months FROM contracts", con)
con.close()

before = sales["revenue"].sum()
print(f"sales lines     {len(sales):>9,}   revenue {before:>14,.2f}")

# The naive version: each sales line picks up its account's term.
naive = sales.merge(contracts, on="account_id", how="left")
after = naive["revenue"].sum()
print(f"after the merge {len(naive):>9,}   revenue {after:>14,.2f}")
print(f"revenue multiplied by {after / before:.2f}, and no error")
print()

# One line of order 44151, priced by hand earlier, after the merge.
one = naive[(naive["order_id"] == 44151)
            & (naive["sku"] == "MRD-SAN-043")]
show = ["order_id", "sku", "revenue", "contract_id", "start_date"]
print(one[show].to_string(index=False))
print()

# Why: the key is not unique on the right-hand side.
per_account = contracts.groupby("account_id").size()
spread = per_account.value_counts().sort_index()
print("contracts per account:", spread.to_dict())
print("contracts.account_id unique?",
      contracts["account_id"].is_unique)
print()

# The fix: make the right side one row per account, then say so.
latest = (contracts.sort_values(["account_id", "start_date"])
          .drop_duplicates("account_id", keep="last"))
fixed = sales.merge(latest, on="account_id", how="left",
                    validate="many_to_one")
total = fixed["revenue"].sum()
print(f"fixed merge     {len(fixed):>9,}   revenue {total:>14,.2f}")

try:
    sales.merge(contracts, on="account_id", how="left",
                validate="many_to_one")
except pd.errors.MergeError as e:
    print("validate= on the naive merge raises MergeError:")
    print(" ", str(e).splitlines()[0].replace("; not", ";\n  not"))

# The numbers, saved for the chapter's figure.
Path(__file__).with_suffix(".json").write_text(json.dumps({
    "lines_before": len(sales), "revenue_before": round(before, 2),
    "lines_after": len(naive), "revenue_after": round(after, 2),
    "lines_fixed": len(fixed), "revenue_fixed": round(total, 2),
    "example": one[["revenue", "contract_id", "start_date"]]
               .to_dict("records"),
    "contracts_per_account": {str(k): int(v)
                              for k, v in spread.items()},
}, indent=1))
