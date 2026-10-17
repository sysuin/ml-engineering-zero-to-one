# Machine Learning Engineering from Zero to One — the code

The companion code for *Machine Learning Engineering from Zero to One* by Sunny Singh: every listing in the
book, **Foresight** — the prediction service it builds across 27 chapters — and the
generator for **Meridian**, the fictional company whose warehouse every example runs
against.

Nothing in the book's output was typed by hand. Each listing here was executed, and its
real output is saved beside it, so you can compare what you get with what the book shows.
Every listing is also deterministic: run it twice and it prints the same thing, to the
last digit.

## Start

```bash
make setup                   # a virtualenv, the libraries, and code/ on Python's path
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python3 code/_preflight.py   # checks every step, in order
```

No GPU, no cloud account and no API key. Appendix B of the book walks through each step
on macOS, Windows, Linux and Colab.

Then the dataset, which is generated rather than downloaded:

```bash
make data                               # seconds, no network, byte-identical every time
```

## Where things are

```
code/NN/            the listings for chapter NN, each with its captured .out
code/foresight/     Foresight, as far as the book has built it
code/meridian/      the dataset generator
tests/              the test suite
docs/               Chapter 3's brief, and the reports the listings write
```

`make help` lists the everyday commands.

## Checking your output against the book

```bash
python3 code/_runner.py 08                  # re-run chapter 8's listings, capture their output
python3 code/_runner.py --check --twice --portable
                                            # every listing runs, and runs the same twice
```

The captures came from an Apple laptop with the library versions in Appendix C. On another
machine or with newer libraries, a last digit can move, and where a listing breaks a tie
between equal values a count can move by a few. `--portable` lists those listings without
failing on them; your run still agrees with itself, which is what matters. CI runs it
nightly on Linux, where about a tenth of the listings differ from the captures in this way.

## Licence

MIT — see [LICENSE](LICENSE). The licence covers this code and the dataset generator, not
the book.
