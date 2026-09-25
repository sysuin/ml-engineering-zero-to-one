"""
Every setting in one place: where the data lives, where models go, and every seed.

Two rules govern this file:

1.  **No chapter hard-codes a path or a seed.** Listings import `SEED`, `DATA` and the
    rest from here, so the whole book moves together when one of them changes, and a
    reader who wants to see how much a result depends on its seed changes one line.

2.  **Every source of randomness is named here.** A result that changes when it is run
    again is not a result. Python's `random`, NumPy, scikit-learn's `random_state`,
    LightGBM and PyTorch each have their own generator, and a listing that seeds four of
    the five is not deterministic. `seed_everything()` seeds all of them;
    `LIGHTGBM_DETERMINISTIC` holds the settings LightGBM needs as well as a seed.

`code/_runner.py` pins what a listing cannot see — thread counts, hash seeds, the time
zone — and `_runner.py --check --twice` runs every listing twice and fails if a single
character differs. This file is the half of that guarantee the reader can see.

No secret appears here or in any file git can see. The one chapter that calls a hosted
language model, Chapter 20, reads its key from the environment, loaded from `.env`.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# The project root: two levels up from this file (code/foresight/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Load .env if python-dotenv is installed. Nothing before Chapter 20 needs it, so its
# absence is not an error. Real environment variables always win.
try:
    from dotenv import load_dotenv
except ImportError:                                           # pragma: no cover
    pass
else:
    load_dotenv(ROOT / ".env", override=False)


# --------------------------------------------------------------------------- seeds
SEED = int(os.getenv("FORESIGHT_SEED", "20260201"))
"""The one seed. Every split, model and resample in the book derives from it."""

MERIDIAN_SEED = 20260101
"""Book 1's dataset seed, fixed in code/meridian/generate.py. Recorded here, never changed:
Book 1's figures depend on its output staying byte-identical."""


def seed_everything(seed: int = SEED) -> None:
    """
    Seed every generator a listing might touch, and ask PyTorch for deterministic kernels.

    Call it once, after the listing's imports. PyTorch is seeded only if the listing has
    imported it: importing it here would add seconds to every listing in Parts I to III,
    which never touch it.
    """
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:                                       # pragma: no cover
        pass
    else:
        np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(THREADS)


def rng(seed: int = SEED):
    """A NumPy Generator. Prefer this to the global state: two listings cannot disturb it."""
    import numpy as np
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- determinism
THREADS = int(os.getenv("FORESIGHT_THREADS", "1"))
"""Threads for model training. One, because summing in a different order changes the last
digit, and the book prints the last digit. Raise it on your own machine for speed, and
expect the fourth decimal place to move."""

LIGHTGBM_DETERMINISTIC = {
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,     # deterministic=True alone still lets LightGBM choose layout
    "num_threads": THREADS,
    "verbose": -1,              # its training chatter is not output the book prints
}
"""Merge into every LightGBM model's parameters: `LGBMClassifier(**LIGHTGBM_DETERMINISTIC, ...)`."""


# --------------------------------------------------------------------------- paths
DATA = ROOT / "data"
"""Everything generated. Not committed: it is rebuilt from a seed in seconds."""

MERIDIAN = DATA / "meridian"
"""Book 1's Meridian, from code/meridian/generate.py, untouched by this book."""

WAREHOUSE = MERIDIAN / "warehouse" / "meridian.db"
"""The SQLite warehouse: regions, suppliers, customers, products, orders."""

MERIDIAN_ML = DATA / "meridian-ml"
"""The scaled dataset for machine learning, from code/meridian/generate_ml.py: 6,058
accounts, labels from a documented process, and the planted traps."""

ML_WAREHOUSE = MERIDIAN_ML / "warehouse" / "meridian_ml.db"
"""The warehouse every model in the book learns from."""

TRUTH = MERIDIAN_ML / "truth"
"""What the generator knows and the warehouse does not: each renewal's true probability of
leaving, and each ticket's true intent. Only evaluation code may read it."""

ARTIFACTS = ROOT / "artifacts"
"""Trained models and their metadata. Chapter 21 turns this into a registry."""

CACHE = DATA / "cache"
"""Training runs over a minute are cached here by the hash of their inputs, so the book
still builds in minutes. Delete it and every number is recomputed, and must come out
the same."""


# --------------------------------------------------------------------------- spending
SPEND_CEILING_USD = float(os.getenv("BOOK_SPEND_CEILING_USD", "2.00"))
"""A rail for Chapter 20's language-model comparison, the only listings that spend money."""
