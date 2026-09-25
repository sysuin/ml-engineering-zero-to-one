# Exercise 1: five more SQL shapes in pandas, each checked in SQLite.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE

con = sqlite3.connect(ML_WAREHOUSE)
sql = lambda q: pd.read_sql(q, con)                     # noqa: E731
accounts = sql("SELECT account_id, region_id, segment, since"
               " FROM accounts")
regions = sql("SELECT region_id, name FROM regions")
tickets = sql("SELECT ticket_id, account_id, opened_at, channel"
              " FROM tickets")


def counted(frame, by):
    """GROUP BY `by` with COUNT(*) AS n, sorted like ORDER BY `by`."""
    return (frame.groupby(by, as_index=False).size()
            .rename(columns={"size": "n"}))


results = {}

# 1. HAVING: segments with more than 600 accounts.
a = sql("SELECT segment, COUNT(*) AS n FROM accounts GROUP BY segment"
        " HAVING COUNT(*) > 600 ORDER BY segment")
b = counted(accounts, "segment").query("n > 600")
results["HAVING"] = (a, b.reset_index(drop=True))

# 2. CASE WHEN: accounts by tenure band.
a = sql("SELECT CASE WHEN since < '2015-01-01' THEN 'before 2015'"
        " ELSE '2015 on' END AS band, COUNT(*) AS n"
        " FROM accounts GROUP BY band ORDER BY band")
early = accounts["since"] < "2015-01-01"
band = early.map({True: "before 2015", False: "2015 on"})
results["CASE WHEN"] = (a, counted(accounts.assign(band=band), "band"))

# 3. JOIN a lookup table: accounts per region name.
a = sql("SELECT r.name, COUNT(*) AS n FROM accounts a JOIN regions r"
        " USING (region_id) GROUP BY r.name ORDER BY r.name")
named = accounts.merge(regions, on="region_id", validate="many_to_one")
results["JOIN"] = (a, counted(named, "name"))

# 4. Anti-join: accounts that never opened a ticket.
a = sql("SELECT COUNT(*) AS n FROM accounts a LEFT JOIN"
        " (SELECT DISTINCT account_id FROM tickets) t"
        " USING (account_id) WHERE t.account_id IS NULL")
never = ~accounts["account_id"].isin(tickets["account_id"])
results["anti-join"] = (a, pd.DataFrame({"n": [never.sum()]}))

# 5. Subquery in WHERE: phone tickets from Enterprise accounts, by year.
a = sql("SELECT substr(opened_at, 1, 4) AS year, COUNT(*) AS n"
        " FROM tickets WHERE channel = 'Phone' AND account_id IN"
        " (SELECT account_id FROM accounts"
        "  WHERE segment = 'Enterprise')"
        " GROUP BY year ORDER BY year")
ent = accounts.loc[accounts["segment"] == "Enterprise", "account_id"]
phone = tickets[(tickets["channel"] == "Phone")
                & tickets["account_id"].isin(ent)]
year = phone["opened_at"].str[:4]
results["IN (subquery)"] = (a, counted(phone.assign(year=year), "year"))
con.close()

for name, (a, b) in results.items():
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print(f"{name:14} {len(a)} rows, identical")
