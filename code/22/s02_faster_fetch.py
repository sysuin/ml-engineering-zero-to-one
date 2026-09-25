# nondeterministic: timing
# Exercise 2: a fetch inside a tighter budget. The same assemble(),
# shown still less of the warehouse: accounts and their history are
# shadowed too. Checked with the skew test before it is timed.
import time
from contextlib import closing

import numpy as np
import pandas as pd

from foresight.config import SEED
from foresight.data.build_table import TABLE
from foresight.pipeline import features
from foresight.pipeline.features import assemble
from foresight.serve import online

NARROWER = """
CREATE TEMP VIEW accounts AS SELECT * FROM main.accounts
    WHERE account_id IN (SELECT account_id FROM scope);
CREATE TEMP VIEW account_history AS SELECT * FROM main.account_history
    WHERE account_id IN (SELECT account_id FROM scope);
"""
KEEP = ["contract_id", "account_id", "end_date", "moment",
        "term_months", "legacy_terms", "discount_pct"]


def narrower(ids, warehouse=None):
    """online.rows(), with two more tables narrowed to the accounts."""
    with closing(online.open_warehouse()) as con:
        found = online.contracts(con, ids)
        online.scope(con, found.account_id)    # needs every account
        con.executescript(NARROWER)
        return assemble(con, found[KEEP])


table = pd.read_parquet(TABLE)
checked = features.check_skew(table, narrower(table.contract_id))
print(f"skew test, {len(table):,} contracts: {checked.differ.sum()}"
      " differences")

sample = table.contract_id.sample(200, random_state=SEED)
for name, fetch in [("online.rows()", online.rows),
                    ("narrower", narrower)]:
    ms = []
    for cid in sample:
        t = time.perf_counter()
        fetch([cid])
        ms.append(1000 * (time.perf_counter() - t))
    q = np.percentile(ms, [50, 95])
    print(f"{name:<15} median {q[0]:5.1f} ms, 95th {q[1]:5.1f} ms")
