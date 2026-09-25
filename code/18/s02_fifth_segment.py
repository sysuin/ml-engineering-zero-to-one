# Exercise 3: the five-segment solution's small-basket group, examined.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.segments import (SEGMENT_DATE, Segmenter,
                                account_features, fit_named, load,
                                profile)

orders, accounts = load()
f = account_features(SEGMENT_DATE, orders, accounts)
four = fit_named(f).label(f)
five = Segmenter(5).fit(f)
labels = pd.Series(five.predict(f), index=f.index)
small = profile(f, labels).basket.idxmin()      # the small baskets
g = f[labels == small]
print(f"Small-basket segment: {len(g)} accounts, median basket "
      f"${g.basket.median():,.0f}, {g.is_key_account.sum()} key")
crm = g.crm_segment.value_counts(normalize=True).head(3)
print("CRM: " + ", ".join(f"{k} {v:.0%}" for k, v in crm.items()))
print("Where they sit among the four segments:")
for k, v in four[labels == small].value_counts().items():
    print(f"  {k:20}{v:>5}")
print(f"Basket under $450: {(g.basket < 450).mean():.0%} of them, "
      f"{(f.basket < 450).mean():.0%} of all accounts")

rows = pd.read_parquet(TABLE).query("end_date <= '2024-12-31'")
parts = []
for moment, cohort in rows.groupby("moment"):
    fm = account_features(moment, orders, accounts)
    hit = pd.Series(five.predict(fm) == small, index=fm.index)
    parts.append(cohort.assign(small=cohort.account_id.map(hit)))
rows = pd.concat(parts).dropna(subset=["small"])
rows["split"] = rows.end_date.le("2024-06-30").map(
    {True: "train", False: "validation"})
by = rows.groupby(["split", "small"]).not_renewed
rate = by.agg(["size", "mean"])
print(f"\n{'':12}{'small-basket':>18}{'everyone else':>18}")
for split in ("train", "validation"):
    s, o = rate.loc[(split, True)], rate.loc[(split, False)]
    print(f"{split:12}{s['size']:>8,.0f}{s['mean']:>10.1%}"
          f"{o['size']:>8,.0f}{o['mean']:>10.1%}")
