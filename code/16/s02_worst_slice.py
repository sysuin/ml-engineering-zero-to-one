# Exercise 2: the largest accounts' leavers were never called. Rank by
# chance times a quarter's revenue instead, and count what each list
# reaches on the validation cohorts.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, calibrated, value_at_stake
from foresight.evaluate import SPLITS, backtest
from foresight.slices import key_accounts, label
from foresight.train import make_model

rows = pd.read_parquet(TABLE)
s = backtest(rows, *SPLITS["validation"], calibrated(make_model()))
s = label(s, rows[rows.end_date.between("2023-01-01", "2024-06-30")],
          key_accounts())
s = s[s.population == "long tail"]
value = value_at_stake(s)
lists = {"by chance (v0.6)": by_capacity(s),
         "by chance x revenue": by_capacity(s.assign(
             model=s.model * value))}
print(f"{'':22}{'leavers':>8}{'largest':>9}{'revenue reached':>17}")
for name, called in lists.items():
    left = called & (s.not_renewed == 1).to_numpy()
    big = left & (s["size"] == "largest").to_numpy()
    print(f"  {name:<20}{left.sum():>8}{big.sum():>9}"
          f"{value[left].sum():>17,.0f}")
big = ((s["size"] == "largest") & (s.not_renewed == 1)).sum()
lost = value[(s.not_renewed == 1).to_numpy()].sum()
print(f"Largest-quarter leavers: {big}; revenue of all leavers:"
      f" {lost:,.0f}")
