# How long each chapter's listings took in the book's last build.
# nondeterministic: reads the timings the runner last recorded
import json
import math
from collections import defaultdict

from foresight.config import ROOT

report = json.loads((ROOT / "code" / "_report.json").read_text())
count, seconds = defaultdict(int), defaultdict(float)
for path, result in report["listings"].items():
    chapter = int(path[:2])
    if chapter <= 27:                    # chapters, not appendices
        count[chapter] += 1
        seconds[chapter] += result["seconds"]

print("One thread, one laptop, as the runner recorded it. A chapter")
print("last built with --twice shows both runs, so read these as")
print("upper bounds for running a chapter once.\n")
print(f"{'chapter':>7}{'listings':>10}{'seconds':>9}  about")
for chapter in sorted(count):
    s = seconds[chapter]
    about = ("under a minute" if s < 60
             else f"{math.ceil(s / 60)} minutes")
    print(f"{chapter:>7}{count[chapter]:>10}{s:>9.0f}  {about}")
total = sum(seconds.values())
print(f"{'all':>7}{sum(count.values()):>10}{total:>9.0f}"
      f"  {math.ceil(total / 60)} minutes")
