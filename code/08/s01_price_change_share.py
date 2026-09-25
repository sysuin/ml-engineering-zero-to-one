# Exercise 1: fit on everything known, or only on what came after July.
import pandas as pd

from foresight.data.build_table import TABLE
from foresight.evaluate import SPLITS, backtest, measure

table = pd.read_parquet(TABLE)
pool = table[table.end_date <= SPLITS["validation"][1]]
CHANGE = pd.Timestamp("2023-07-01")


class AfterOnly:
    """Chapter 7's model, fitted only on contracts marked after the
    small-business price change."""

    def fit(self, rows):
        from foresight.models.logistic import RenewalRisk
        self.model = RenewalRisk().fit(rows[rows.moment >= CHANGE])
        return self

    def predict_proba(self, rows):
        return self.model.predict_proba(rows)


# The 2024 cohorts: contracts ending April to December 2024.
first, last = "2024-04-01", SPLITS["validation"][1]
print(f"{'fitted on outcomes known at the mark':<38}{'AUC':>7}"
      f"{'top-40 leavers':>16}")
for name, make in (("all of them", None), ("marked after 1 July 2023",
                                            AfterOnly)):
    scored = (backtest(pool, first, last) if make is None
              else backtest(pool, first, last, make))
    m = measure(scored)
    hits = round(m["precision"]["model"] * 40 * scored.moment.nunique())
    print(f"  {name:<36}{m['auc']['model']:>7.3f}{hits:>16}")
print(f"{scored.moment.nunique()} cohorts, {len(scored):,} contracts,"
      f" {scored.not_renewed.sum()} leavers")
