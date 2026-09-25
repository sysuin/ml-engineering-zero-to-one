# One Meridian order in plain Python: values, lists and dicts.
import sqlite3

from foresight.config import ML_WAREHOUSE

ORDER_ID = 44151                    # an integer: a whole number
REGION = "Midwest"                  # a string: text, in quotes
KEY_ACCOUNT = True                  # a boolean: True or False

con = sqlite3.connect(ML_WAREHOUSE)
con.row_factory = sqlite3.Row       # rows that know their column names
rows = con.execute(
    "SELECT line_no, sku, qty, unit_price, discount_pct"
    " FROM order_lines WHERE order_id = ? ORDER BY line_no",
    (ORDER_ID,),
).fetchall()
con.close()

lines = [dict(r) for r in rows]     # a list of dicts: one per row
print(f"order {ORDER_ID} has {len(lines)} lines")
print("first row, as a dict:")
for name, value in lines[0].items():
    print(f"  {name:13} {value!r}")

qty = [line["qty"] for line in lines]   # one column, as a list
print("qty column:", qty)


def line_revenue(line):
    """Revenue for one order line, after its discount."""
    discount = line["discount_pct"] / 100
    return line["qty"] * line["unit_price"] * (1 - discount)


total = 0.0
for line in lines:                  # a formula dragged down the rows
    total = total + line_revenue(line)
print(f"order total after discounts: {total:,.2f}")
print("type of each value:", type(ORDER_ID).__name__,
      type(REGION).__name__, type(KEY_ACCOUNT).__name__,
      type(total).__name__)
