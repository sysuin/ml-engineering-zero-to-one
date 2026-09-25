# The job's settings: read from configs/renewal.toml, changed for one
# run from the command line, and refused when a key is misspelt or a
# value is the wrong type.
from foresight.pipeline import settings

cfg = settings.load("renewal")
for section in ("model", "data"):
    for key, value in cfg[section].items():
        print(f"{section + '.' + key:<26}{value}")

once = settings.load("renewal", ["data.as_of=2024-12-01",
                                 "evaluation.cohorts=3"])
print(f"\nfor one run: as of {once['data']['as_of']},"
      f" {once['evaluation']['cohorts']} cohorts\n(the file still says"
      f" {cfg['data']['as_of']},"
      f" {cfg['evaluation']['cohorts']} cohorts)")

for bad in ("data.as-of=2024-12-01", "model.strenght=0.01",
            "evaluation.cohorts=six"):
    try:
        settings.load("renewal", [bad])
    except (KeyError, ValueError) as e:
        print(f"{bad:<26}refused: {type(e).__name__}")
try:
    settings.load("renewals")
except FileNotFoundError as e:
    print(f"\n{e}")
