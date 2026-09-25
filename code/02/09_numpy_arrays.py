# From a table of accounts to a grid of numbers: what a model wants.
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
table = pd.read_sql("""
    SELECT account_id, segment,
           COUNT(DISTINCT order_id) AS orders_2024,
           SUM(revenue)             AS revenue_2024,
           julianday('2025-01-01')
             - julianday(MAX(order_date)) AS days_since
    FROM   v_sales
    WHERE  order_date >= '2024-01-01' AND order_date < '2025-01-01'
    GROUP  BY account_id, segment
    ORDER  BY account_id
""", con, index_col="account_id")
con.close()

X = table[["orders_2024", "revenue_2024", "days_since"]].to_numpy()
print("X is a", type(X).__name__, "of", X.dtype,
      "with shape", X.shape)
print("row for account 62:  ", X[table.index.get_loc(62)])
print("first column, 5 rows:", X[:5, 0])
print()

means, stds = X.mean(axis=0), X.std(axis=0)   # one per column
Z = (X - means) / stds                        # broadcasting
print("column means:   ", np.round(means, 1))
print("scaled means:   ", np.round(Z.mean(axis=0), 6) + 0.0)  # no -0.
print("scaled stds:    ", np.round(Z.std(axis=0), 6))
print()

mixed = table[["segment", "orders_2024"]].to_numpy()
print("with a text column the grid becomes", mixed.dtype,
      "->", mixed[0])
try:
    mixed.mean(axis=0)
except TypeError as e:
    print("and .mean() fails:", type(e).__name__)
