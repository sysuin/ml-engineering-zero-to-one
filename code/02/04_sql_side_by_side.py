# Two SQL queries and their pandas twins, proved to give the same rows.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)

# --- 1. SELECT, WHERE, ORDER BY, LIMIT --------------------------------
sql_top = pd.read_sql("""
    SELECT order_id, account, sku, revenue
    FROM   v_sales
    WHERE  order_date >= '2025-01-01' AND region = 'Midwest'
    ORDER  BY revenue DESC, order_id
    LIMIT  5
""", con)

sales = pd.read_sql("SELECT * FROM v_sales"
                    " WHERE order_date >= '2025-01-01'", con)
mask = sales["region"] == "Midwest"          # WHERE: a column of bools
columns = ["order_id", "account", "sku", "revenue"]       # SELECT
pd_top = (sales.loc[mask, columns]
          .sort_values(["revenue", "order_id"],
                       ascending=[False, True])           # ORDER BY
          .head(5)                                        # LIMIT
          .reset_index(drop=True))

# --- 2. GROUP BY with three aggregates --------------------------------
sql_regions = pd.read_sql("""
    SELECT region,
           COUNT(DISTINCT order_id) AS orders,
           SUM(revenue)             AS revenue,
           AVG(revenue)             AS avg_line
    FROM   v_sales
    WHERE  order_date >= '2025-01-01'
    GROUP  BY region
    ORDER  BY revenue DESC
""", con)
con.close()

pd_regions = (sales.groupby("region", as_index=False)
              .agg(orders=("order_id", "nunique"),
                   revenue=("revenue", "sum"),
                   avg_line=("revenue", "mean"))
              .sort_values("revenue", ascending=False)
              .reset_index(drop=True))

print(pd_top.to_string(index=False))
print()
print(pd_regions.round(2).to_string(index=False))
print()
for name, a, b in [("top five lines", sql_top, pd_top),
                   ("region totals", sql_regions, pd_regions)]:
    pd.testing.assert_frame_equal(a, b)      # raises if they differ
    print(f"{name}: SQL and pandas agree, and .equals() says "
          f"{a.equals(b)}")
