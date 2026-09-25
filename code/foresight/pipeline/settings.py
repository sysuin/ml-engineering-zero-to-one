"""
The training job's settings, kept in a file beside the code rather
than in it. Chapter 21 writes it.

    configs/renewal.toml       the renewal model's settings

A setting belongs in the file when changing it should not need a
change of code: the model's strength, the dates, how many cohorts the
backtest scores, where things are written. load() reads the file,
applies any overrides given on the command line as section.key=value,
and refuses a key the file does not have, so that a misspelt setting
is an error rather than a setting quietly ignored.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

CONFIGS = Path(__file__).parent / "configs"


def path(name: str) -> Path:
    """The settings file for model `name`."""
    p = CONFIGS / f"{name}.toml"
    if not p.exists():
        known = ", ".join(sorted(f.stem
                                 for f in CONFIGS.glob("*.toml")))
        raise FileNotFoundError(f"no settings for model {name!r};"
                                f" known: {known}")
    return p


def parse(value: str, like):
    """`value` read as the same type as the setting it replaces."""
    if isinstance(like, bool):
        if value not in ("true", "false"):
            raise ValueError(f"expected true or false, not {value!r}")
        return value == "true"
    return type(like)(value)


def load(name: str, overrides=()) -> dict:
    """The settings for `name`, with overrides applied."""
    settings = tomllib.loads(path(name).read_text())
    settings = copy.deepcopy(settings)
    for item in overrides:
        key, sep, value = item.partition("=")
        section, _, field = key.partition(".")
        if (not sep or section not in settings
                or field not in settings[section]):
            raise KeyError(f"unknown setting {key!r}")
        settings[section][field] = parse(value,
                                         settings[section][field])
    return settings
