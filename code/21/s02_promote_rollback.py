# Exercise 2: two runs of the training job, a month apart, into one
# registry; the later one promoted over the earlier, then rolled back
# with the registry's command, which prints the registry.
# timeout: 300
import tempfile
from pathlib import Path

from foresight.pipeline import registry
from foresight.pipeline.run import run

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for as_of in ("2024-11-01", "2024-12-01"):
        run("renewal", [f"data.as_of={as_of}"], root=root,
            say=lambda line: None)
    reg = registry.Registry("renewal", root / "artifacts")
    reg.promote(1, "November's list")
    reg.promote(2, "December's list")
    registry.main(["renewal", "rollback", "December's list queried",
                   "--root", str(root / "artifacts")])
    model, manifest = reg.production()
    runs = (root / "mlruns" / "runs.jsonl").read_text().splitlines()
    as_of = manifest["data"]["as_of"]
    print(f"\nin production: the model fitted on {as_of}"
          f"\nruns in the log: {len(runs)}; versions in the folder:"
          f" {len(reg.read()['versions'])}")
