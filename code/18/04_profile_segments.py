# Four segments, profiled by the medians a sales director can check.
import json

import pandas as pd

from foresight.segments import (SEGMENT_DATE, Segmenter,
                                account_features, fit_named, load,
                                profile)

orders, accounts = load()
f = account_features(SEGMENT_DATE, orders, accounts)
seg = fit_named(f, k=4)
names = seg.label(f)
prof = profile(f, names).sort_values("tenure_years")

print(f"Medians on {SEGMENT_DATE}")
print(f"{'':19}{'orders':>7}{'basket':>8}{'days':>6}{'years':>7}"
      f"{'trend':>7}{'spend':>9}")
for name, r in prof.iterrows():
    print(f"{name:19}{r.orders_365:>7.0f}{r.basket:>8,.0f}"
          f"{r.recency_days:>6.0f}{r.tenure_years:>7.1f}"
          f"{r.trend:>+7.2f}{r.spend_365:>9,.0f}")
print(f"{'all accounts':19}{f.orders_365.median():>7.0f}"
      f"{f.basket.median():>8,.0f}{f.recency_days.median():>6.0f}"
      f"{f.tenure_years.median():>7.1f}{f.trend.median():>+7.2f}"
      f"{f.spend_365.median():>9,.0f}")

crm = pd.crosstab(names, f.crm_segment, normalize="index")
key = f.groupby(names).is_key_account.sum()
print(f"\n{'':19}{'accounts':>9}{'share':>7}{'of spend':>10}"
      f"{'key':>5}{'small biz':>11}")
for name, r in prof.iterrows():
    print(f"{name:19}{r.accounts:>9,.0f}{r.share:>7.0%}"
          f"{r.spend_share:>10.0%}{key[name]:>5}"
          f"{crm.loc[name, 'Small business']:>11.0%}")
print(f"{'all accounts':19}{len(f):>9,}{1:>7.0%}{1:>10.0%}"
      f"{f.is_key_account.sum():>5}"
      f"{(f.crm_segment == 'Small business').mean():>11.0%}")

print(f"\nThe other candidates{'accounts':>9}{'orders':>7}{'basket':>8}"
      f"{'days':>6}{'years':>7}{'trend':>7}")
for k in (3, 5):
    other = Segmenter(k).fit(f)
    p = profile(f, other.predict(f)).sort_values("tenure_years")
    for i, (_, r) in enumerate(p.iterrows()):
        name = f"k = {k}" if i == 0 else ""
        print(f"{name:19}{r.accounts:>9,.0f}"
              f"{r.orders_365:>7.0f}{r.basket:>8,.0f}"
              f"{r.recency_days:>6.0f}{r.tenure_years:>7.1f}"
              f"{r.trend:>+7.2f}")

json.dump({"date": SEGMENT_DATE,
           "profile": prof.reset_index().round(4).to_dict("records"),
           "overall": {c: float(f[c].median()) for c in
                       ["orders_365", "basket", "recency_days",
                        "tenure_years", "trend", "spend_365"]},
           "key": {k: int(v) for k, v in key.items()},
           "small_business": crm["Small business"].round(4).to_dict()},
          open("code/18/04_profile_segments.json", "w"), indent=1)
