# Exercise 3: delete everything from the moment on; nothing may change.
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import build

MOMENT, END = "2024-01-01", "2024-03-31"    # one cohort's mark, end
TICKETS = """
    SELECT c.contract_id,
           (SELECT COUNT(*) FROM tickets t
             WHERE t.account_id = c.account_id
               AND t.opened_at >= date(c.end_date, '-180 days')
               AND t.opened_at < c.end_date) AS tickets_90d
    FROM contracts c WHERE c.end_date = ? AND c.outcome IS NOT NULL"""


def cut_warehouse(path: Path) -> None:
    """A copy of the warehouse, every record from the moment on gone."""
    con = sqlite3.connect(path)
    con.execute("ATTACH ? AS w", (str(ML_WAREHOUSE),))
    keep = {"regions": "1", "accounts": "1",
            "contracts": f"end_date = '{END}'",   # the rows and labels
            "orders": f"order_date < '{MOMENT}'",
            "order_lines": "order_id IN (SELECT order_id FROM orders)",
            "tickets": f"opened_at < '{MOMENT}'",
            "account_history": f"valid_from < '{MOMENT}'"}
    for table, where in keep.items():
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM w.{table}"
                    f" WHERE {where}")
    con.commit()
    con.close()


with tempfile.TemporaryDirectory() as tmp:
    cut = Path(tmp) / "cut.db"
    cut_warehouse(cut)
    for name, run in (
            ("the colleague's tickets_90d",
             lambda db: pd.read_sql_query(TICKETS, sqlite3.connect(db),
                                          params=(END,))),
            ("Foresight's builder, every column",
             lambda db: build(END, END, warehouse=db))):
        full, past = run(ML_WAREHOUSE), run(cut)
        same = full.equals(past)
        rows = (full != past).any(axis=1).sum() if not same else 0
        print(f"{name:34} unchanged: {same!s:5}  rows moved: {rows}")
