# Exercise 4: resolving the legacy ids without the start date.
import sqlite3

import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import legacy_ids, tidy

con = sqlite3.connect(ML_WAREHOUSE)
pairs = pd.read_sql_query("""
    SELECT old.account_id AS legacy_id, new.account_id,
           old.name AS legacy_name, new.name
    FROM accounts old JOIN accounts new
      ON new.postcode = old.postcode AND new.crm_source = 'Meridian CRM'
    WHERE old.crm_source = 'Legacy CRM'""", con)
named = pairs[pairs.legacy_name.map(tidy) == pairs.name.map(tidy)]
per_legacy = named.groupby("legacy_id").account_id.nunique()
print(f"Postcode and tidied name: {len(named)} pairs,"
      f" {named.legacy_id.nunique()} legacy ids,"
      f" {named.account_id.nunique()} current ids")
print("  legacy ids with more than one match:"
      f" {(per_legacy > 1).sum()}")
same = dict(zip(named.legacy_id, named.account_id)) == legacy_ids(con)
print(f"  the same mapping as legacy_ids(): {same}")
