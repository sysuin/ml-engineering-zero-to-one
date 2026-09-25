# The full-year line's weights, and three reasons not to trust them yet.
import numpy as np
from sklearn.linear_model import LinearRegression

from foresight.data.spend import (FEATURES, TRAIN, VALIDATION, features,
                                  mae, spend_table, year_on_record)

train = spend_table(*TRAIN)
train = train[year_on_record(train)]
valid = spend_table(*VALIDATION)
X, y = features(train), train.spend_next_90d.to_numpy()
model = LinearRegression().fit(X, y)
per_sd = model.coef_ * X.std(axis=0)            # one std dev's worth

print(f"{'feature':<18}{'weight':>12}{'per unit':>11}"
      f"{'per std dev':>13}")
units = ["dollar", "dollar", "order", "order", "day", "ticket", "day",
         "month", "yes"]
for i in np.argsort(-abs(per_sd)):
    print(f"{FEATURES[i]:<18}{model.coef_[i]:>12,.3f}{units[i]:>11}"
          f"{per_sd[i]:>13,.0f}")

# Reason 2: columns that move together share their weight unstably.
cols = [0, 1, 2, 3]
corr = np.corrcoef(np.column_stack([X[:, cols], y]).T)
print(f"\n{'correlation':<18}" + "".join(
    f"{h:>9}" for h in ("s_90d", "s_365", "o_90d", "o_prev", "target")))
for i, c in enumerate(cols):
    print(f"{FEATURES[c]:<18}" + "".join(
        f"{corr[i, j]:>9.2f}" for j in range(len(cols) + 1)))
drop = LinearRegression().fit(np.delete(X, 0, axis=1), y)
print(f"\n{'weight':<18}{'with spend_90d':>16}{'without':>12}")
for c, w in zip(cols[1:], drop.coef_[:3]):
    print(f"{FEATURES[c]:<18}{model.coef_[c]:>16,.3f}{w:>12,.3f}")
v = valid.spend_next_90d
with_it = mae(v, model.predict(features(valid)))
without = mae(v, drop.predict(np.delete(features(valid), 0, axis=1)))
print(f"{'validation MAE':<18}{with_it:>16,.0f}{without:>12,.0f}")

# Reason 3: tickets arrive with size, and size is not in the weight.
print("\nby tickets in the 90 days before the mark")
print(f"  {'tickets':<9}{'rows':>7}{'mean spend_365':>16}"
      f"{'mean target':>13}")
bucket = np.minimum(train.tickets_90d, 2)
for k, g in train.groupby(bucket):
    print(f"  {['0', '1', '2+'][k]:<9}{len(g):>7,}"
          f"{g.spend_365.mean():>16,.0f}"
          f"{g.spend_next_90d.mean():>13,.0f}")
