# Five systems sized on the back of an envelope, before any boxes.
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from foresight.config import ML_WAREHOUSE

DAY = 86_400                 # seconds
BYTES = 8                    # one stored feature value
HEADROOM = 0.5               # run cores half loaded, to survive a spike
NS_PER_CELL = 10             # assumed: boosting costs about 10 ns of
                             # one core per row, per feature, per tree


@dataclass
class Workload:
    name: str
    requests_day: float      # decisions asked for, per day
    scored_per_req: int      # rows the model scores for each request
    peak_ratio: float        # busiest hour's rate over the day's mean
    cpu_ms: float            # model time for one scored row, one core
    wait_ms: float           # time one request spends in the system
    features: int
    entities: float          # keys held in an online feature store
    logged_per_req: int      # scored rows whose features are kept
    train_rows: float        # rows in one training table
    retrains_month: int
    window_h: float = 0      # a batch job's window; 0 means online
    trees: int = 500

    def estimate(self) -> dict:
        rows_day = self.requests_day * self.scored_per_req
        if self.window_h:              # batch: finish in the window
            per_s = rows_day / (self.window_h * 3600)
            in_flight = None
        else:                          # online: survive the peak
            req_s = self.requests_day / DAY * self.peak_ratio
            per_s = req_s * self.scored_per_req
            in_flight = req_s * self.wait_ms / 1000   # Little's law
        cells = self.train_rows * self.features * self.trees
        return {
            "rows scored a day": rows_day,
            "rows a second": per_s,
            "requests in flight": in_flight,
            "cores to score": per_s * self.cpu_ms / 1000 / HEADROOM,
            "online store, GB": self.entities * self.features
            * BYTES * 2 / 1e9,           # x2 for keys and indexes
            "training table, GB": self.train_rows * self.features
            * BYTES / 1e9,
            "feature log, GB/yr": self.requests_day
            * self.logged_per_req * 365
            * (self.features * BYTES + 32) / 1e9,
            "retrain, core-h/mo": cells * NS_PER_CELL / 1e9 / 3600
            * self.retrains_month}


def show(x) -> str:
    """Three significant figures, with k, M or G for large numbers."""
    if x is None:
        return "-"
    if 0 < abs(x) < 0.01:
        return "<0.01"
    for unit, size in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= size:
            return f"{x / size:.3g}{unit}"
    return f"{x:.3g}"


def main():
    # Foresight is measured: its renewals, from the warehouse. The rest
    # are assumptions, written down so that they can be argued with.
    con = sqlite3.connect(ML_WAREHOUSE)
    rows, months = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT substr(end_date, 1, 7))
        FROM contracts WHERE outcome IS NOT NULL
          AND end_date BETWEEN '2023-01-01' AND '2024-06-30'
        """).fetchone()
    cards, fraud, keep = 8e6, 0.001, 0.01   # a day; rate; legit kept
    # name, requests a day, rows scored each, peak ratio, CPU ms, wait
    # ms, features, online keys, rows logged, training rows, retrains
    systems = [
        Workload("Foresight", rows / months, 1, 1, 0.05, 0, 40, 0, 1,
                 rows, 1, window_h=1),     # a whole cohort, one day
        Workload("Fraud", cards, 1, 4, 0.5, 40, 300, 15e6, 1,
                 90 * cards * (fraud + keep), 4),
        Workload("Recs", 30e6, 500, 3, 0.02, 80, 150, 21e6, 10,
                 30e6 * 10 * 7 * 0.2, 30),
        Workload("Search", 2e6, 200, 4, 0.02, 150, 120, 5e6, 20,
                 2e6 * 20 * 60 * 0.1, 4),
        Workload("Demand", 24e6, 28, 1, 0.02, 0, 80, 0, 1, 1e9, 1,
                 window_h=4),
    ]
    table = {s.name: s.estimate() for s in systems}
    print(f"Foresight: {rows:,} renewals in {months} monthly cohorts\n")
    print(f"{'':19}" + "".join(f"{n:>9}" for n in table))
    for metric in table["Foresight"]:
        print(f"{metric:19}" + "".join(
            f"{show(t[metric]):>9}" for t in table.values()))
    Path("code/26/01_estimates.json").write_text(json.dumps(
        {"systems": {s.name: s.__dict__ for s in systems},
         "estimates": table}, indent=1))


if __name__ == "__main__":
    main()
