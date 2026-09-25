# DuckDB: SQL over DataFrames in memory, with a DataFrame back.
import sqlite3

import duckdb
import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
sales = pd.read_sql("SELECT order_date, account_id, region, sku,"
                    " revenue FROM v_sales"
                    " WHERE order_date >= '2024-01-01'", con,
                    parse_dates=["order_date"])
con.close()

# DuckDB finds `sales` by its Python name.
# QUALIFY filters on a window function, as HAVING does on a SUM.
top_sku = duckdb.sql("""
    SELECT region, sku, SUM(revenue) AS revenue
    FROM   sales
    WHERE  order_date >= DATE '2025-01-01'
    GROUP  BY region, sku
    QUALIFY ROW_NUMBER() OVER (PARTITION BY region
                               ORDER BY SUM(revenue) DESC) = 1
    ORDER  BY region
""").df()
print(top_sku.round(2).to_string(index=False))
print()

# Days since each account's last order on 1 January 2025.
recency = duckdb.sql("""
    SELECT account_id,
           date_diff('day', MAX(order_date), DATE '2025-01-01')
               AS days_since
    FROM   sales
    WHERE  order_date < DATE '2025-01-01'
    GROUP  BY account_id
""").df()
print(type(recency).__name__, recency.shape)
print(recency["days_since"].describe().round(1).to_string())
