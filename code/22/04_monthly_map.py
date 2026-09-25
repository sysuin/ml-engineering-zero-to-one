# The map refitted, the weights kept: v0.6 as registered at go-live on
# 1 January 2025, making the list marked 1 September 2025, with its
# Platt map as fitted and as the monthly job refits it.
# timeout: 300
import shutil
import tempfile
from pathlib import Path

from foresight.pipeline.registry import Registry
from foresight.pipeline.run import run
from foresight.serve import batch, renewal

root = Path(tempfile.mkdtemp())
done = run("renewal", ["data.as_of=2025-01-01"], root=root,
           say=lambda *lines: None)
Registry("renewal", root / "artifacts").promote(done["version"],
                                                "go-live")
made = batch.month("2025-09-01", root=root)
model, manifest, _ = batch.production(root, "2025-09-01")
print(f"cohort {made['mark']:%Y-%m-%d}; model fitted on"
      f" {manifest['data']['rows']:,} outcomes to"
      f" {manifest['data']['as_of']}")
print(f"Platt's map: {made['calibration']}")

scored = made["scored"]
tail = scored[scored.is_key_account == 0]
rows = tail[list(manifest["inputs"]) + ["moment", "end_date"]]
before = model.predict_proba(rows)[:, 1]
refit, _ = renewal.recalibrate(
    model, renewal.known(manifest, "2025-09-01"),
    manifest["data"]["as_of"])
print(f"\n{'':22}{'slope':>7}{'shift':>8}{'mean chance':>13}")
for name, m, p in [("as fitted", model, before),
                   ("refitted", refit, tail.chance)]:
    print(f"  {name:<20}{m.map_.a_:>7.2f}{m.map_.b_:>8.2f}"
          f"{p.mean():>13.1%}")
order = tail.assign(before=before).sort_values(
    ["before", "contract_id"], ascending=[False, True])
same = (order.contract_id.head(40).to_numpy()
        == tail.contract_id.head(40).to_numpy()).sum()
print(f"\nthe forty, in order: {same} of 40 the same either way")
shutil.rmtree(root)
