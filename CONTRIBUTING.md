# Foresight — notes for whoever works on this next

This file is for anyone about to change this repository. It says the things that are
true about this repository but not visible in any one file.

## What this is

Foresight is Meridian's prediction service — a fictional mid-size distributor — and the
worked project of *Machine Learning Engineering from Zero to One*, Book 2 of the Zero to One
Series. It answers three questions: which accounts are about to leave, how much will sell
next month, and which support ticket needs a human first. It is built across 27 chapters
in 10 milestones, from a one-page brief (v0.1, Chapter 3) to a deployed, monitored service
that Book 1's Clarity can call as a tool (v1.0, Chapter 27).

As Foresight grows, each milestone's code is kept as the book showed it. **Do not refactor
a milestone's code to match a later one.** It is not duplication to be cleaned up; it is the
illustration. If a bug exists in the v0.3 model, it exists in Chapter 7 too, and fixing one
without the other breaks the book.

## Where things are

```
code/foresight/         the project; config.py holds every seed and path
code/meridian/          the dataset generator, carried over from Book 1 byte for byte
code/NN/                the executed listings for chapter NN, with their captured .out
code/_runner.py         runs every listing, captures its output, proves it repeats
tests/                  the test suite, which grows with Foresight
```

## The rules this repository is built on

**No output in the book was typed by a human.** `code/_runner.py` executes every listing
and captures its real stdout. If you change a listing, re-run it. If you change a module
a listing imports, *touch the listing* — the cache keys on the listing file's hash, not
its imports, so a shared-module change will not re-run its dependents on its own.

**Every listing is deterministic.** Every source of randomness is seeded from
`foresight.config.SEED`; LightGBM runs with `LIGHTGBM_DETERMINISTIC`; PyTorch uses
deterministic algorithms on CPU; thread counts are pinned by the runner. `make verify`
runs every listing twice and fails if one character differs. A listing that cannot be
deterministic — a wall-clock timing, a hosted model — says so on its first lines with
`# nondeterministic: <reason>`, and its prose must survive any value it prints.

**No metric without its baseline.** "AUC 0.81" never appears alone in prose; "AUC 0.81
against 0.64 for the last-order-date rule" does. The baseline is computed by the same
listing, on the same split.

**Prose that states a direction or a magnitude must compute it.** A sentence saying "the
forest beat the tree" is a bug waiting for the next run to flip. Derive it in the listing
and print the sentence, or write the prose so it survives either outcome. This is the
single most common way a chapter goes wrong.

**No library version or price appears in a chapter's prose.** They live in
`requirements.txt` and Appendix C.

**Book 1's dataset does not change.** `code/meridian/` is Book 1's generator, and its
output must stay byte-identical, because Book 1 quotes it. The scaled dataset this book
needs comes from a second generator that reads the first's output and never writes to it
(`generate_ml.py`, checked by `verify_ml.py`).

**No secret appears in any file git can see.** The one key the book ever uses, for
Chapter 20's language-model comparison, comes from `.env`, which is ignored.

## Running it

`make help` lists everything. The four you will use:

```
make data                generate Meridian from its seed, and verify it
make listings            run every listing that changed
make verify              nothing stale, and every listing prints the same thing twice
```

## What to be careful about

- **The planted traps are verified, not assumed.** A trap that `verify_ml.py` does not
  prove present is a chapter whose lesson may not happen.
