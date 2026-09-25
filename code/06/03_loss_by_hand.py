# A loss by hand on three accounts, then what each loss punishes.
import numpy as np

from foresight.data.spend import (TRAIN, VALIDATION, mae, rmse,
                                  spend_table)

train = spend_table(*TRAIN)
three = train.set_index("contract_id").loc[[7332, 7335, 7326]]
x = (three.spend_90d / 1000).round(2).to_numpy()      # thousands
y = (three.spend_next_90d / 1000).round(2).to_numpy()

w, b = 1.0, 0.0                   # the last-value rule, as a line
pred = w * x + b
error = pred - y
print(f"{'contract':>8}{'x':>7}{'y':>7}{'pred':>7}{'error':>8}"
      f"{'|error|':>9}{'error^2':>9}")
for c, *v in zip(three.index, x, y, pred, error, abs(error),
                 error ** 2):
    print(f"{c:>8}" + "".join(f"{n:>{k}.2f}" for n, k in
                              zip(v, (7, 7, 7, 8, 9, 9))))
print(f"MAE  {np.mean(abs(error)):.3f}   MSE {np.mean(error ** 2):.3f}"
      f"   RMSE {np.sqrt(np.mean(error ** 2)):.3f}   ($ thousands)")
print(f"gradient at w = 1, b = 0:   dw {2 * np.mean(error * x):.3f}"
      f"   db {2 * np.mean(error):.3f}")

# The best single number under each loss, found by trying them all.
t = train.spend_next_90d.to_numpy()
grid = np.linspace(0, 10_000, 10_001)          # every whole dollar
best_abs = grid[np.argmin([np.mean(abs(t - c)) for c in grid])]
best_sq = grid[np.argmin([np.mean((t - c) ** 2) for c in grid])]
print(f"\nbest constant for absolute error {best_abs:>8,.0f}"
      f"   median {np.median(t):>8,.0f}")
print(f"best constant for squared error  {best_sq:>8,.0f}"
      f"   mean   {np.mean(t):>8,.0f}")

valid = spend_table(*VALIDATION)
v = valid.spend_next_90d.to_numpy()
print(f"\nOn validation{'':<23}{'MAE':>10}{'RMSE':>10}")
for name, c in (("the median", np.median(t)), ("the mean", t.mean())):
    print(f"  {name:<34}{mae(v, c):>10,.0f}{rmse(v, c):>10,.0f}")

# Who the misses belong to: the last-value rule's errors, split.
miss = v - valid.spend_90d.to_numpy()
key = valid.is_key_account.to_numpy() == 1
print(f"\n{key.sum()} key-account rows of {len(v):,} ({key.mean():.1%})"
      " carry, of the last-value rule's")
print(f"  absolute error  {abs(miss[key]).sum() / abs(miss).sum():.1%}")
sq = miss ** 2
print(f"  squared error   {sq[key].sum() / sq.sum():.1%}")
