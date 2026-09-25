"""
Foresight — Meridian's prediction service, built across this book.

Foresight answers the three questions Meridian's operators actually ask:

1. Which accounts are about to leave?        renewal-risk classification, Parts I–III
2. How much will we sell next month?         demand forecasting, Part IV
3. Which support ticket needs a human first? ticket triage from text, Part IV

It grows in ten milestones, from a one-page brief in Chapter 3 (v0.1, a document, not
code) to a deployed, monitored service in Chapter 27 (v1.0) that Clarity, the assistant
from *AI Engineering from Zero to One*, can call as a tool. Each chapter that adds to it
leaves this package working.

`config.py` holds the seeds, paths and determinism settings every listing imports, so that
no chapter hard-codes a path or a random seed of its own. Each milestone adds modules
beside it and never rewrites one an earlier chapter printed; `__version__` is the latest
milestone the package completes.
"""

__version__ = "0.2"
