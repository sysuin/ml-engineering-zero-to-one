# v0.5's training job, twice: once with today's account manager added
# to the model's columns, once as it ships.
# timeout: 300
import sqlite3

import pandas as pd

from foresight.checks.leakage import LeakageError
from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.features.sources import load_sources
from foresight.train import train

sources = load_sources()
table = pd.read_parquet(TABLE)
now = pd.read_sql_query("SELECT account_id, account_manager FROM"
                        " accounts", sqlite3.connect(ML_WAREHOUSE))
leaky = table.merge(now, on="account_id")
leaky["desk_now"] = (leaky.account_manager == "Retention desk") * 1.0

try:
    train("2025-01-01", leaky, extra=("desk_now",), sources=sources)
except LeakageError as e:
    print("With desk_now:", str(e).split("\n\n")[0], "\n")

run = train("2025-01-01", sources=sources)
print("As it ships:")
print(run["report"].text())
print("\nfor a person to review:")
for line in run["report"].reviews():
    print(f"  {line}")
print(f"\nFitted on {run['rows']:,} outcomes known on {run['as_of']};"
      f" ready to\nscore the {run['recent']} contracts marked"
      f" {run['mark']:%Y-%m-%d}")
