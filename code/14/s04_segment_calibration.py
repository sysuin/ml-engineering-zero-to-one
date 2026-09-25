# Exercise 4: does calibrating the whole list calibrate each segment?
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import calibrated
from foresight.evaluate import SPLITS, backtest
from foresight.models.regularised import maker

table = pd.read_parquet(TABLE)
raw = backtest(table, *SPLITS["validation"], maker("l1", 0.002))
cal = backtest(table, *SPLITS["validation"],
               calibrated(maker("l1", 0.002)))
raw["platt"] = cal.model.to_numpy()
print(f"{'segment':<16}{'contracts':>10}{'left':>7}{'as fitted':>11}"
      f"{'Platt':>8}")
for seg, c in raw.groupby("segment"):
    print(f"{seg:<16}{len(c):>10,}{c.not_renewed.mean():>7.1%}"
          f"{c.model.mean():>11.1%}{c.platt.mean():>8.1%}")
