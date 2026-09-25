# Foresight's ten milestones as its own record kept them: each line is
# read from a page or an output an earlier chapter's listing wrote.
import json
import re

import pandas as pd

from foresight.config import ROOT


def page(path, pattern):
    """The first match of `pattern` in what an earlier listing wrote."""
    text = (ROOT / path).read_text()
    return re.search(pattern, text, re.S).groups()


def saved(path):
    return json.loads((ROOT / path).read_text())


brief = saved("code/03/03_baseline_rule.json")
(rows, rules) = page("code/05/09_expectations.out",
                     r"([\d,]+) rows, (\d+) expectations hold")
(val,) = page("docs/foresight-evaluation.md",
              r"precision at 40\s+(\S+ points \(.*?\))")
(test3,) = page("docs/foresight-v0.3-test.md",
                r"precision at 40\s+(\S+ points \(.*?\))")
board = saved("code/11/11_leaderboard.json")
hits = {m: round(v["precision"][0] * board["calls"])
        for m, v in board["lists"].items()}
v05 = saved("code/13/12_test_report.json")
v06 = saved("code/16/15_test_report.json")
wape = page("code/17/07_global_model.out",
            r"seasonal naive\s+(\S+%).*?model \+ long tail\s+(\S+%)")
(days,) = page("code/18/10_daily_job.out", r"alerted on (\d+ of \d+)")
urgent = page("code/20/15_foresight_v08.out",
              r"urgent words\s+(\S+)\s+model\s+(\S+)")
job = page("code/21/07_train_command.out",
           r"(\d+) leavers in (\d+) calls \(rule (\d+)\)")
tonight = saved("code/22/03_monthly_job.json")
march = "2025-03-02"
notices = saved("code/24/04_notices.json")[march]
seen = pd.Timestamp(march) + pd.Timedelta(days=31)  # notices in
year = saved("code/24/12_champion_challenger.json")
impact = saved("code/25/11_impact_report.json")["retention"]

record = [
    ("0.1", 3, "the brief", f"the rule's 40 calls:"
     f" {brief['rule_precision']:.1%} leavers (random"
     f" {brief['base_rate']:.1%}, perfect"
     f" {brief['perfect_precision']:.1%})"),
    ("0.2", 5, "the training table", f"{rows} rows, {rules}"
     " expectations checked on every build"),
    ("0.3", 8, "logistic, backtested", f"model - rule, validation"
     f" {val}; 2025 {test3}"),
    ("0.4", 11, "four models", f"lasso {hits['lasso']}, booster"
     f" {hits['boosting']}, rule {hits['rule']} of"
     f" {board['calls']} calls; ceiling {board['ceiling']}"),
    ("0.5", 13, "features, leak checks", f"no group helped; 2025:"
     f" {v05['hits']['v0.5']} of {v06['calls_total']}"
     f" (ceiling {v05['ceiling']})"),
    ("0.6", 16, "calibrated, explained", f"2025: {v06['hits']} vs"
     f" rule {v06['rule_hits']}; log loss {v06['raw_log_loss']}"
     f" -> {v06['log_loss']}"),
    ("0.7", 18, "forecast, alerts", f"WAPE {wape[1]} vs {wape[0]};"
     f" Pemberton flagged on {days} days"),
    ("0.8", 20, "ticket triage", f"Urgent in the first 3 a day:"
     f" {urgent[1]} vs {urgent[0]} (keywords)"),
    ("0.9", 23, "pipeline, service", f"make train: {job[0]} of"
     f" {job[1]} (rule {job[2]}); the job: {tonight['contracts']}"
     f" scored, {tonight['held']} held out"),
    ("1.0", 27, "monitored, measured", f"Voss seen {seen.day} {seen:%B}"
     f" ({notices['noticed']:.0f} notices vs"
     f" {notices['expected']:.1f}); served"
     f" {year['hits']['as served']}"
     f" vs rule {year['hits']['rule']} of {year['calls']}; calls"
     f" +{100 * impact['diff']:.1f} pts (simulated)"),
]
for version, chapter, what, measured in record:
    print(f"v{version:<4}Ch {chapter:<3}{what}")
    words, line = measured.split(), ""
    for w in words:                     # wrap at the page's width
        if len(line) + len(w) + 1 > 58:
            print(f"{'':10}{line}")
            line = ""
        line = f"{line} {w}".strip()
    print(f"{'':10}{line}")

with open("code/27/01_the_record.json", "w") as f:
    json.dump([{"version": v, "chapter": c, "what": w}
               for v, c, w, _ in record], f)
