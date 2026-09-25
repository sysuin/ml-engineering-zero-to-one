"""
A model registry in a folder: every artifact a training job has made,
and which one is in production. Chapter 21 writes it.

    python -m foresight.pipeline.registry renewal
    python -m foresight.pipeline.registry renewal promote 2 "why"
    python -m foresight.pipeline.registry renewal rollback "why"

    artifacts/renewal/
        registry.json       each version's stage, and a log of changes
        1/  2/  3/ ...      one artifact a version, never changed

A version arrives staged. promote() makes a staged version production
and archives the one it replaces; rollback() puts back the version
that production last replaced and archives the current one. Nothing
is deleted and nothing in a version's folder is rewritten. Every
change is appended to the log with the reason given for it, so the
registry can say which model made any list, and why it was the one.
One process writes at a time: a team that trains from two machines
at once needs a registry with locks, such as MLflow's.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from foresight.config import ARTIFACTS
from foresight.pipeline import artifact

STAGES = ("staged", "production", "archived")


class RegistryError(Exception):
    """A change the registry will not make."""


class Registry:
    def __init__(self, model: str, root: Path = ARTIFACTS):
        self.folder = Path(root) / model
        self.file = self.folder / "registry.json"

    # ---------------------------------------------- the file
    def read(self) -> dict:
        if not self.file.exists():
            return {"versions": {}, "log": []}
        return json.loads(self.file.read_text())

    def _write(self, state: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(self.file)          # all of the change, or none

    def _move(self, state, version: str, stage: str, reason: str):
        state["versions"][version]["stage"] = stage
        state["log"].append({
            "version": int(version), "stage": stage, "reason": reason,
            "when": datetime.now(timezone.utc).isoformat(
                timespec="seconds")})

    # ---------------------------------------------- changes
    def register(self, model, manifest: dict, card: str | None = None,
                 reason: str = "trained") -> int:
        """Save a new artifact as the next version, staged."""
        state = self.read()
        version = str(max(map(int, state["versions"]), default=0) + 1)
        saved = artifact.save(self.folder / version, model, manifest,
                              card)
        state["versions"][version] = {
            "run": saved["run"], "as_of": saved["data"]["as_of"],
            "metrics": saved["metrics"], "stage": None}
        self._move(state, version, "staged", reason)
        self._write(state)
        return int(version)

    def promote(self, version: int, reason: str) -> None:
        """Make a staged version production; archive the old one."""
        state, v = self.read(), str(version)
        if v not in state["versions"]:
            raise RegistryError(f"no version {version}")
        if state["versions"][v]["stage"] != "staged":
            raise RegistryError(f"version {version} is"
                                f" {state['versions'][v]['stage']},"
                                " not staged")
        artifact.verify(self.folder / v)
        for old in self._in("production", state):
            self._move(state, old, "archived", f"replaced by {v}")
        self._move(state, v, "production", reason)
        self._write(state)

    def rollback(self, reason: str) -> int:
        """Return the version production last replaced."""
        state = self.read()
        now = self._in("production", state)
        replaced = [str(e["version"]) for e in state["log"]
                    if e["reason"].startswith("replaced by")]
        back = [v for v in reversed(replaced)
                if state["versions"][v]["stage"] == "archived"]
        if not now or not back:
            raise RegistryError("nothing to roll back to")
        back = back[0]
        artifact.verify(self.folder / back)
        self._move(state, now[0], "archived", f"rolled back: {reason}")
        self._move(state, back, "production", f"restored: {reason}")
        self._write(state)
        return int(back)

    # ---------------------------------------------- questions
    def _in(self, stage: str, state=None) -> list[str]:
        state = state or self.read()
        return [v for v, r in state["versions"].items()
                if r["stage"] == stage]

    def production(self):
        """The production model and its manifest, verified."""
        now = self._in("production")
        if not now:
            raise RegistryError("no version is in production")
        return artifact.load(self.folder / now[0])

    def page(self) -> str:
        """Every version on a line, then the log, no wider than 68."""
        state = self.read()
        out = [f"{'version':>7}  {'stage':<12}{'as of':<12}"
               f"{'leavers':>8}{'AUC':>7}"]
        for v, r in sorted(state["versions"].items(),
                           key=lambda kv: int(kv[0])):
            m = r["metrics"]
            out.append(f"{v:>7}  {r['stage']:<12}{r['as_of']:<12}"
                       f"{m['model hits']:>8}{m['model auc']:>7.3f}")
        out += ["", f"{'log':>7}"]
        for e in state["log"]:
            out.append(f"{e['version']:>7}  {e['stage']:<12}"
                       f"{e['reason']}")
        return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("action", nargs="?", default="list",
                    choices=["list", "promote", "rollback"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--root", type=Path, default=ARTIFACTS)
    a = ap.parse_args(argv)
    reg = Registry(a.model, a.root)
    if a.action == "promote":
        version, *why = a.args
        reg.promote(int(version), " ".join(why) or "promoted")
    elif a.action == "rollback":
        reg.rollback(" ".join(a.args) or "rolled back")
    print(reg.page())


if __name__ == "__main__":
    main()
