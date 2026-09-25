"""
The register every feature is entered in. Chapter 12 writes it.

A feature is a name, a group, the day it entered the library, the
sources it reads, how far back it looks, how a linear model should see
it, one sentence saying what it means, and the function that computes
it. The function is given a Context (sources.py) and returns one value
per contract, computed from records dated strictly before the mark.

    @feature("order_trend", "trends", reads=("orders",), window=180,
             about="log of (orders, last 90 days + 1) over (the 90 "
                   "days before + 1)")
    def order_trend(ctx): ...

A name can be registered once. Chapter 4's columns keep their one
definition in build_table.py, and no feature here may reuse one of
their names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from foresight.data.build_table import COLUMNS

ADDED = "2026-09-25"            # the day Chapter 12 wrote the library


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    added: str                  # the day the definition was written
    reads: tuple[str, ...]      # sources: orders, lines, tickets, ...
    window: int                 # days of history it looks back; 0: none
    linear: str                 # "raw" or "log": for linear models
    about: str                  # what it means, in one sentence
    compute: Callable


REGISTRY: dict[str, Feature] = {}


def feature(name: str, group: str, reads: tuple = (), window: int = 0,
            about: str = "", linear: str = "raw", added: str = ADDED):
    """Register the decorated function as the one definition of
    `name`. A second definition of the same name is an error."""
    if name in REGISTRY or name in COLUMNS:
        raise ValueError(f"feature {name!r} is already defined")
    if linear not in ("raw", "log"):
        raise ValueError(f"{name}: linear must be 'raw' or 'log'")

    def register(fn: Callable) -> Callable:
        REGISTRY[name] = Feature(name, group, added, tuple(reads),
                                 window, linear, about, fn)
        return fn
    return register


def names(groups=None) -> list[str]:
    """Every feature's name, or those in the given groups, in the
    order they were registered."""
    if isinstance(groups, str):
        groups = (groups,)
    return [n for n, f in REGISTRY.items()
            if groups is None or f.group in groups]


def groups() -> list[str]:
    """The groups, in the order their first feature was registered."""
    return list(dict.fromkeys(f.group for f in REGISTRY.values()))
