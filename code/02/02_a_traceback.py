# expect-fail
# A misspelt column name, and the traceback Python prints for it.


def line_revenue(line):
    price = line["unit_prce"]
    return line["qty"] * price * (1 - line["discount_pct"] / 100)


line = {"line_no": 1, "sku": "MRD-SAN-043", "qty": 15,
        "unit_price": 56.53, "discount_pct": 0}
print("about to price one line")
print(line_revenue(line))
