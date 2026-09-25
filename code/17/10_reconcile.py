# Products, categories, regions and the total: four ways to make the
# forecasts add up, scored at every level on 2025.
import json

import pandas as pd

from foresight.forecast.backtest import backtest, wape
from foresight.forecast.baselines import (naive, seasonal_naive,
                                          seasonal_smoothing)
from foresight.forecast.model import backtest_model
from foresight.forecast.reconcile import (LEVELS, aggregate,
                                          proportional, top_down)
from foresight.forecast.series import (CATEGORY_REGION, PRODUCT_REGION,
                                       monthly_units)

year = pd.period_range("2025-01", "2025-12", freq="M")
units = monthly_units(PRODUCT_REGION)
panels = {name: aggregate(units, by) if name != "product x region"
          else units for name, by in LEVELS.items()}


def wide(r, names):
    w = r.pivot_table(index=["h", "target"], columns="series",
                      values="forecast")
    if len(names) > 1:
        w.columns = pd.MultiIndex.from_tuples(w.columns, names=names)
    return w


def model(by):
    r = backtest_model(monthly_units(by), (monthly_units(by, "tail"),),
                       targets=year)
    return wide(r, by)


def rule(f, name):
    return wide(backtest(panels[name], f, targets=year),
                LEVELS[name] or ["total"])


products, categories = model(PRODUCT_REGION), model(CATEGORY_REGION)
total = rule(seasonal_smoothing, "total")["total"]
shares = []
for h, target in total.index:                 # the last 12 months
    last = units.loc[target - h - 11:target - h].fillna(0).sum()
    shares.append(last / last.sum())
shares = pd.DataFrame(shares, index=total.index)


def last_year(history, h):          # Sanitation had no 2024 Q1:
    return seasonal_naive(history, h).fillna(naive(history, h))


methods = {
    "seasonal naive": rule(last_year, "product x region"),
    "bottom-up": products,
    "top-down": top_down(total, shares),
    "middle-out": proportional(products, categories, CATEGORY_REGION)}
alone = [products, categories, rule(seasonal_smoothing, "region"),
         total.to_frame("total")]


def score(forecast, name):
    actual = panels[name].loc[[t for _, t in forecast.index],
                              forecast.columns]
    return wape(actual.to_numpy().ravel(), forecast.to_numpy().ravel())


def up(bottom):
    """A bottom-level forecast summed to every level."""
    return [aggregate(bottom, by) if name != "product x region"
            else bottom for name, by in LEVELS.items()]


heads = ["product", "category", "region", "total"]
print(f"{'WAPE, 2025, h=1-3':<18}"
      + "".join(f"{x:>10}" for x in heads))
print(f"{'':<18}" + f"{'x region':>10}" * 2)
for label, frames in [("each level alone", alone)] + [
        (k, up(b)) for k, b in methods.items()]:
    print(f"  {label:<16}" + "".join(
        f"{score(f, n):>10.1%}" for f, n in zip(frames, LEVELS)))
gap = aggregate(products, CATEGORY_REGION) - categories
print(f"\nCategory x region: the two models disagree by "
      f"{gap.abs().sum().sum() / categories.sum().sum():.1%} of units")

# One month, h=1, for the figure: the total, the regions, the Midwest.
key = (1, pd.Period("2025-10", freq="M"))
forecasts = {"alone": alone, "bottom-up": up(products),
             "middle-out": up(methods["middle-out"])}
tree = {}
for i, name in enumerate(LEVELS):
    if name == "product x region":
        continue
    cols = [c for c in alone[i].columns
            if name != "category x region" or c[1] == "Midwest"]
    tree[name] = {str(c): {k: round(float(v[i].loc[key, c]))
                           for k, v in forecasts.items()}
                  | {"actual": float(panels[name].loc[key[1], c])}
                  for c in cols}
with open("code/17/10_reconcile.json", "w") as f:
    json.dump({"month": str(key[1]), "tree": tree}, f)
