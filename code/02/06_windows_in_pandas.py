# SQL window functions and their pandas twins.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

MONTHLY = """
    SELECT region, substr(order_date, 1, 7) AS month,
           SUM(revenue) AS revenue
    FROM   v_sales
    WHERE  order_date >= '2024-01-01'
    GROUP  BY region, month
"""
con = sqlite3.connect(ML_WAREHOUSE)
sql = pd.read_sql(f"""
    WITH m AS ({MONTHLY})
    SELECT region, month, revenue,
           revenue / SUM(revenue) OVER (PARTITION BY month) AS share,
           AVG(revenue) OVER (PARTITION BY region ORDER BY month
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS avg_3m,
           LAG(revenue) OVER (PARTITION BY region
                              ORDER BY month)                AS prev
    FROM   m
    ORDER  BY region, month
""", con)
monthly = pd.read_sql(MONTHLY, con).sort_values(["region", "month"])
con.close()

by_region = monthly.groupby("region")["revenue"]
month_total = monthly.groupby("month")["revenue"].transform("sum")
monthly["share"] = monthly["revenue"] / month_total
monthly["avg_3m"] = by_region.transform(
    lambda s: s.rolling(3, min_periods=1).mean())
monthly["prev"] = by_region.shift(1)
monthly = monthly.reset_index(drop=True)

pd.testing.assert_frame_equal(sql, monthly)
print("SQL windows and pandas agree on all", len(monthly),
      "region-months\n")

midwest = monthly[monthly["region"] == "Midwest"].set_index("month")
view = midwest[["revenue", "share", "avg_3m", "prev"]].head(5)
view["change"] = view["revenue"] / view["prev"] - 1
money = "{:,.0f}".format
print(view.to_string(formatters={
    "revenue": money, "avg_3m": money, "prev": money,
    "share": "{:.3f}".format, "change": "{:+.1%}".format}))

strict = midwest["revenue"].rolling(3).mean().head(3)
print("\nrolling(3) with no min_periods:",
      ", ".join("NaN" if pd.isna(v) else money(v) for v in strict))
