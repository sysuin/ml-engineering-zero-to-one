"""
Foresight's impact report: one page, for a finance audience. Chapter
25 writes it.

It says what each of Foresight's three predictions changed, in the
units finance counts (accounts kept and dollars, stock dollars, hours),
each against the thing it replaced and with a range. It says which
numbers are measured and which rest on an assumption, and a page
filled from a simulation says so in its first line.

    render(findings) -> str      the page, as text no wider than 68
    write(findings, path)        the page, as a Markdown file
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from foresight.config import ROOT
from foresight.impact.analyse import dollars

PAGE = ROOT / "docs" / "foresight-impact-report.md"
CALLS_A_YEAR = 480              # forty a cohort, twelve cohorts


def _k(v: float) -> str:
    """Dollars to the nearest thousand."""
    return dollars(round(v, -3))


def render(f: dict) -> str:
    r, s, t, d = f["retention"], f["stock"], f["triage"], f["design"]
    lines = [f"Foresight: what it changed in {f['year']}",
             "One page for finance. Every range is a 95% interval"
             " unless", "it says otherwise.", ""]
    if f.get("simulated"):
        lines += [
            "SIMULATED. No retention calls are on record. The"
            " retention",
            "lines come from a simulation in which a call keeps"
            f" {f['assumed_save']:.0%} of",
            "would-be leavers (the brief's assumption). They show"
            " what the",
            "test will report, not what it has found.", ""]
    v, lo, hi = r["per_call"]
    lines += [
        f"Retention calls: {r['cohorts']} cohorts; of each top 40,"
        f" {r['n'] // r['cohorts']} called,",
        f"{r['n'] // r['cohorts']} held out at random",
        f"  Left, held out / called  {r['held']:.1%} /"
        f" {r['held'] - r['diff']:.1%}",
        f"  Calls cut leaving by     {100 * r['diff']:.1f} points"
        f" ({100 * r['lo']:.1f} to {100 * r['hi']:.1f})",
        f"  Save rate                {r['save_rate']:.0%} of"
        f" would-be leavers ({r['save_lo']:.0%} to"
        f" {r['save_hi']:.0%})",
        f"  Net per call             {dollars(v)} ({dollars(lo)} to"
        f" {dollars(hi)})",
        f"  A year of {CALLS_A_YEAR} calls      "
        f"{_k(v * CALLS_A_YEAR)} ({_k(lo * CALLS_A_YEAR)} to"
        f" {_k(hi * CALLS_A_YEAR)})",
        f"  Caution: {r['cohorts']} of the {d['cohorts']} cohorts"
        f" planned. At a {f['assumed_save']:.0%} save rate",
        f"  a year this size finds the effect"
        f" {d['power_one_year']:.0%} of the time, and",
        "  overstates it when it does.", ""]
    lines += [
        f"Stock: ordering to the forecast, {s['series']} product x"
        " region series",
        f"  Point forecast against last year's month   "
        f"{_k(s['point'])} a year",
        f"  Range forecast against a padded last year  "
        f"{_k(s['range'])} a year",
        f"  As the cost assumptions move: {_k(s['low'])} to"
        f" {_k(s['high'])}", ""]
    lines += [f"Urgent tickets: {t['n']} in {f['year']}, hours to first"
              " read", f"  {'':<22}{'median':>8}{'90% within':>12}"]
    for name, waits in t["waits"].items():
        w = np.asarray(waits)
        lines.append(f"  {name:<22}{np.median(w):>8.1f}"
                     f"{np.quantile(w, 0.9):>12.1f}")
    lines += [f"  Assumed desk: {t['desk']}", ""]
    lines += ["Not claimed here",
              "  Accounts kept by the forecast or by triage; the"
              " retention",
              "  effect on accounts outside the top 40; any effect"
              " of calls",
              "  on accounts that were staying anyway.", "",
              "Asked of finance"] + [f"  {a}" for a in f["ask"]]
    return "\n".join(lines)


def write(findings: dict, path: Path = PAGE) -> str:
    """Write the page as Markdown; return its text."""
    text = render(findings)
    title, body = text.split("\n", 1)
    Path(path).write_text(
        f"# {title}\n\n```text\n{body.lstrip()}\n```\n")
    return text
