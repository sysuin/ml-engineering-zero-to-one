# A straight line fitted to a yes-or-no label, and where it breaks.
import json

from sklearn.linear_model import LinearRegression

from foresight.models.logistic import TRAIN, VALIDATION, features, load

train, valid = load(*TRAIN), load(*VALIDATION)
y = train.not_renewed.to_numpy()
print(f"Training rows {len(train):,}, not renewed {y.sum()}"
      f" ({y.mean():.1%})")

# One column: days since the account's last order, at the mark.
gap = features(train)[["days_since_order"]].to_numpy()
line = LinearRegression().fit(gap, y)
w, b = line.coef_[0], line.intercept_
print(f"\nThe line: {b:.4f} + {w:.5f} x days since the last order")
print(f"  {'days':<10}{'rows':>6}{'left':>8}{'the line says':>15}")
bands = [(1, 7), (8, 14), (15, 30), (31, 60), (61, 90), (91, 180),
         (181, 411)]
for lo, hi in bands:
    inside = (gap[:, 0] >= lo) & (gap[:, 0] <= hi)
    said = b + w * gap[inside, 0].mean()
    print(f"  {f'{lo}-{hi}':<10}{inside.sum():>6}"
          f"{y[inside].mean():>8.1%}{said:>15.1%}")
print(f"The line passes 100% at {(1 - b) / w:,.0f} days")

# Every column the table offers, as numbers.
X, Xv = features(train), features(valid)
many = LinearRegression().fit(X, y)
print(f"\nThe line through all {X.shape[1]} columns")
for name, grid in (("training", X), ("validation", Xv)):
    p = many.predict(grid)
    print(f"  {name:<11} below 0: {(p < 0).sum():>4}   above 1:"
          f" {(p > 1).sum():>2}   lowest {p.min():+.3f}")
low = train.loc[many.predict(X).argmin()]
print(f"  lowest: contract {low.contract_id}, {low.segment},"
      f" {low.orders_90d} orders in 90 days")

# For the figure: every training row's gap and outcome.
with open("code/07/01_line_fails.json", "w") as f:
    json.dump({"w": w, "b": b, "gap": gap[:, 0].tolist(),
               "left": y.tolist()}, f)
