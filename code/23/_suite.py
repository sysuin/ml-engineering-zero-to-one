"""
Run pytest from Python and keep what happened to each test: its file,
its layer, whether it is slow, and how it ended. Chapter 23's listings
use it so that they can print a summary that is the same on every run,
where pytest's own last line carries a time that is not.
"""
from __future__ import annotations

import contextlib
import io
import linecache

import pandas as pd
import pytest

LAYERS = ("unit", "data", "model", "service")


class _Record:
    def __init__(self):
        self.tests: dict[str, dict] = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        for item in items:
            layer = [m for m in LAYERS if item.get_closest_marker(m)]
            own = [m.name for m in item.own_markers if m.name in LAYERS]
            self.tests[item.nodeid] = {
                "test": item.nodeid, "file": item.module.__name__,
                "name": item.originalname, "layer": (own or layer)[0],
                "slow": item.get_closest_marker("slow") is not None,
                "outcome": "collected", "message": "", "line": ""}

    def pytest_runtest_logreport(self, report):
        t = self.tests.get(report.nodeid)
        if t is None or t["outcome"] == "failed":
            return
        if hasattr(report, "wasxfail"):
            t["outcome"] = "xfailed" if report.skipped else "xpassed"
        elif report.failed:
            t["outcome"] = "failed"
            crash = getattr(report.longrepr, "reprcrash", None)
            if crash:
                t["message"] = crash.message.splitlines()[0]
                t["line"] = linecache.getline(crash.path,
                                              crash.lineno).strip()
        elif report.skipped:
            t["outcome"] = "skipped"
        elif report.when == "call":
            t["outcome"] = "passed"


def run(*args: str) -> pd.DataFrame:
    """pytest with `args`, quietly; one row per test it selected."""
    rec = _Record()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        pytest.main([*args, "-q", "-p", "no:cacheprovider"],
                    plugins=[rec])
    columns = ["test", "file", "name", "layer", "slow", "outcome",
               "message", "line"]
    return pd.DataFrame(list(rec.tests.values()), columns=columns)
