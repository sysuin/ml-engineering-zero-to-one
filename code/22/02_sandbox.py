# A second installation of Foresight to work in: the training job run
# as it would be on the morning of 31 December 2025, and a person's
# promotion; then Chapter 20's triage model saved and promoted too.
# timeout: 300
import shutil

from foresight.pipeline.registry import Registry
from foresight.pipeline.run import run
from foresight.serve import SANDBOX, triage

shutil.rmtree(SANDBOX, ignore_errors=True)          # start clean
done = run("renewal", ["data.as_of=2025-12-31"], root=SANDBOX,
           say=lambda *lines: None)
renewals = Registry("renewal", SANDBOX / "artifacts")
renewals.promote(done["version"], "the list for 31 December")

version = triage.train(SANDBOX)
tickets = Registry("triage", SANDBOX / "artifacts")
tickets.promote(version, "Chapter 20's model, v0.8")

for reg, v, what in [(renewals, done["version"], "outcomes known on"),
                     (tickets, version, "tickets opened before")]:
    _, m = reg.production()
    print(f"{reg.folder.name:<8} version {v} in production\n"
          f"{'':9}fitted on {m['data']['rows']:,} {what}"
          f" {m['data']['as_of']}\n"
          f"{'':9}reads {len(m['inputs'])} column(s)")
files = sorted(str(p.relative_to(SANDBOX)) for p in
               (SANDBOX / "artifacts").rglob("*") if p.is_file())
print("\n" + "\n".join(files))
