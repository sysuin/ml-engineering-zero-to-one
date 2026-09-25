"""
Foresight's feature library (Chapter 12): named, dated, point-in-time
definitions, one per feature, each tested.

    registry.py     Feature, the @feature decorator, REGISTRY, names()
    sources.py      the warehouse read once, and Context.window(), the
                    only way a feature reaches an event
    definitions.py  every feature, grouped as Chapter 12 builds them
    encoding.py     target encoding, naive and out of fold
    build.py        build() for any rows; load(), the cached table
"""
from foresight.features import definitions  # noqa: E402,F401  registers
