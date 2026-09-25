# A notebook's three cells, as a script others can run and trust.
import argparse
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE


def load_sales(year: int) -> pd.DataFrame:
    """One row per order line in `year`, from the warehouse."""
    con = sqlite3.connect(ML_WAREHOUSE)
    try:
        return pd.read_sql(
            "SELECT order_date, region, revenue FROM v_sales"
            " WHERE order_date >= ? AND order_date < ?", con,
            params=(f"{year}-01-01", f"{year + 1}-01-01"),
            parse_dates=["order_date"])
    finally:
        con.close()                 # runs even if the query fails


def quarterly_revenue(sales: pd.DataFrame, region: str) -> pd.Series:
    """Revenue by calendar quarter for one region."""
    mine = sales[sales["region"] == region]
    if mine.empty:
        raise ValueError(f"no sales for region {region!r}")
    by_date = mine.set_index("order_date")["revenue"]
    return by_date.resample("QS").sum()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quarterly revenue for one region.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--region", default="Midwest")
    args = parser.parse_args()

    quarters = quarterly_revenue(load_sales(args.year), args.region)
    print(f"{args.region} revenue by quarter, {args.year}")
    for start, revenue in quarters.items():
        print(f"  Q{start.quarter}  {revenue:>14,.2f}")


if __name__ == "__main__":          # run as a script, not on import
    main()
