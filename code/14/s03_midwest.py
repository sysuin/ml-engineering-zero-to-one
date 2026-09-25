# Exercise 3: the Midwest's revenue at risk, contract by contract.
import sqlite3

import numpy as np
import pandas as pd

from foresight.config import ML_WAREHOUSE
from foresight.data.build_table import TABLE
from foresight.decide import calibrated, simulate, value_at_stake
from foresight.evaluate import SPLITS, backtest, interval
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
s = backtest(table, *SPLITS["validation"],
             calibrated(maker("l1", 0.002)))
s["value"] = value_at_stake(s)
key = pd.read_sql_query("SELECT account_id, is_key_account"
                        " FROM accounts",
                        sqlite3.connect(ML_WAREHOUSE))
s = s.merge(key)
mw = s[(s.region == "Midwest") & (s.is_key_account == 0)]
risk = mw.model * mw.value
print(f"Midwest, long tail: {len(mw)} contracts,"
      f" {mw.not_renewed.sum()} left")
print(f"  at risk ${risk.sum():,.0f}; walked"
      f" ${(mw.not_renewed * mw.value).sum():,.0f}")
share = risk.sort_values(ascending=False).cumsum() / risk.sum()
print(f"  the largest 20 contracts carry {share.iloc[19]:.0%} of it")
draws = simulate(mw.model, mw.value, np.zeros(len(mw)))[0]
lo, hi = interval(draws)
print(f"  95% range ${lo:,.0f} to ${hi:,.0f}")
lost = mw.value[mw.not_renewed == 1].sort_values(ascending=False)
print(f"  largest leaver ${lost.iloc[0]:,.0f};"
      f" expected leavers {mw.model.sum():.1f}")
big = mw.value >= mw.value.median()
for name, part in (("larger half", mw[big]),
                   ("smaller half", mw[~big])):
    print(f"  {name:<13} expected {part.model.sum():>5.1f} leavers,"
          f" {part.not_renewed.sum():>3} left")
