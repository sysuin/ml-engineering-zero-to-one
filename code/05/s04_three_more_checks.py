# Exercise 4: three more findings, written as expectations.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.data.expectations import Expectation


def cohort_size(t):
    sizes = t.groupby("moment").size()
    return sizes.between(150, 400)


def since(t):
    return t.moment - pd.to_timedelta(t.tenure_days, unit="D")


MORE = [
    Expectation("every monthly cohort holds 150 to 400 renewals",
                cohort_size),
    Expectation("legacy terms only for customers from before 2019",
                lambda t: (t.legacy_terms == 0)
                | (since(t) < "2019-01-01")),
    Expectation("customers start on the first of a month",
                lambda t: since(t).dt.day == 1),
]

table = pd.read_parquet(TABLE)
broken = table.assign(tenure_days=table.tenure_days + 1)   # off by one
half = table.iloc[::2]                                     # rows lost
for name, t in [("the table", table), ("tenure off by one", broken),
                ("half the rows", half)]:
    print(name)
    for e in MORE:
        holds = e.holds(t)
        print(f"  {e.rule:50}{(~holds).sum():>6,} fail")

sizes = table.groupby("moment").size()
odd = sizes[~sizes.between(150, 400)]
for moment, n in odd.items():
    print(f"The cohort outside the bounds: {moment:%Y-%m-%d}, {n} rows")
