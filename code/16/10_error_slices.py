# v0.6 on the validation cohorts, slice by slice: the leavers its lists
# missed, the renewers they called, and whether its chances held, each
# with a 95% interval.
import json

import pandas as pd

from foresight.data.build_table import TABLE
from foresight.decide import by_capacity, calibrated
from foresight.evaluate import SPLITS, backtest
from foresight.slices import SLICES, key_accounts, label, page, table
from foresight.train import make_model

rows = pd.read_parquet(TABLE)
s = backtest(rows, *SPLITS["validation"], calibrated(make_model()))
train = rows[rows.end_date.between("2023-01-01", "2024-06-30")]
s = label(s, train, key_accounts())
called = by_capacity(s)
print(page(s, called))

with open("code/16/10_error_slices.json", "w") as f:
    json.dump({by: table(s, by, called).round(5).reset_index()
               .to_dict("records") for by in SLICES}, f)
