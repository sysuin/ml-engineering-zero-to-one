# The training job as one command, run into an empty folder: what
# `make train MODEL=renewal` does, step by step, then the card it wrote
# beside Chapter 16's and the registry it left.
# timeout: 300
import tempfile
from pathlib import Path

from foresight.card import CARD
from foresight.pipeline.run import run

with tempfile.TemporaryDirectory() as tmp:
    done = run("renewal", root=Path(tmp))

    card = done["card"].splitlines()
    ch16 = CARD.read_text().splitlines()
    test = ch16.index("## Test year, read once for v0.6")
    same = card == ch16[:test - 1]
    word = "the same as" if same else "different from"
    print(f"\nThe card: {len(card)} lines, {word} Chapter 16's line for"
          f" line\nas far as its test year's section")
    print(f"\n{done['registry'].page()}")
    files = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*")
                   if p.is_file() and "mlruns/artifacts" not in str(p))
    print("\n" + "\n".join(files))
