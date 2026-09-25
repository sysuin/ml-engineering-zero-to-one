# Five charts, five questions, drawn with matplotlib to a file.
import json
import sqlite3
from pathlib import Path

import matplotlib
import matplotlib.dates
import numpy as np
import pandas as pd

matplotlib.use("Agg")               # draw to a file, never a window
import matplotlib.pyplot as plt     # noqa: E402

from foresight.config import DATA, ML_WAREHOUSE, ROOT  # noqa: E402

con = sqlite3.connect(ML_WAREHOUSE)
orders = pd.read_sql("""
    SELECT order_id, order_date, account_id, segment, region,
           SUM(revenue) AS value
    FROM   v_sales
    GROUP  BY order_id
""", con, parse_dates=["order_date"])
con.close()

by_date = orders.set_index("order_date")["value"]
monthly = by_date.resample("MS").sum() / 1e6
y24 = orders[orders["order_date"].dt.year == 2024]
y25 = orders[orders["order_date"].dt.year == 2025]
regions = y25.groupby("region")["value"].sum().sort_values() / 1e6
pair = pd.concat({"y2024": y24.groupby("account_id")["value"].sum(),
                  "y2025": y25.groupby("account_id")["value"].sum()},
                 axis=1).dropna()           # accounts buying in both
segments = ["Small business", "Mid-market", "Public sector",
            "Enterprise"]
by_segment = [y25.loc[y25["segment"] == s, "value"] for s in segments]

fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].plot(monthly.index, monthly.values)
ax[0].xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax[0].set_title("Line: how has it changed?")
ax[1].barh(regions.index, regions.values)
ax[1].set_title("Bar: which is biggest?")
ax[2].hist(y25["value"], bins=np.arange(0, 3001, 100))
ax[2].set_title("Histogram: how is it spread?")
ax[3].scatter(pair["y2024"], pair["y2025"], s=3)
ax[3].set(xscale="log", yscale="log",
          title="Scatter: do two things move together?")
ax[4].boxplot(by_segment, showfliers=False)
ax[4].set_xticks(range(1, 5), [s.split()[0] for s in segments])
ax[4].set_title("Box: how do groups compare?")
fig.tight_layout()

out = DATA / "charts" / "five_charts.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
plt.close(fig)
print("saved", out.relative_to(ROOT))
print()

r = np.corrcoef(np.log(pair["y2024"]), np.log(pair["y2025"]))[0, 1]
first, last = monthly.index[0], monthly.index[-1]
print(f"line       monthly revenue {monthly[first]:.1f}M in"
      f" {first:%b %Y}, {monthly[last]:.1f}M in {last:%b %Y}")
print(f"bar        largest region in 2025: {regions.index[-1]},"
      f" {regions.iloc[-1]:.1f}M")
print(f"histogram  order value: median {y25['value'].median():,.0f},"
      f" mean {y25['value'].mean():,.0f}")
print(f"scatter    {len(pair):,} accounts,"
      f" log revenue 2024 vs 2025: r = {r:.2f}")
print("box        median order value in 2025 by segment:")
for s, v in zip(segments, by_segment):
    print(f"           {s:15} {v.median():,.0f}")

# Compact versions of the five datasets, for the chapter's figure.
hist, edges = np.histogram(y25["value"], bins=np.arange(0, 3001, 100))
thin = pair.sort_index().iloc[::8]
quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
Path(__file__).with_suffix(".json").write_text(json.dumps({
    "line": [round(v, 3) for v in monthly.tolist()],
    "bar": {k: round(v, 2) for k, v in regions.items()},
    "histogram": hist.tolist(),
    "scatter": thin.round(0).values.tolist(),
    "box": {s: [round(float(v.quantile(q)), 0) for q in quantiles]
            for s, v in zip(segments, by_segment)},
}))
