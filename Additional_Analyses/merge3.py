"""
Merge the per-household outputs of run3.py (tags <tag>/H3, <tag>/H4, <tag>/H6)
into <tag>/.

  python merge3.py v3_tree dev
  python merge3.py v3_tree test
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FINAL3_ROOT", os.path.join(HERE, "final3"))
tag, stage = sys.argv[1], sys.argv[2]
D = os.path.join(ROOT, tag)
parts = [os.path.join(D, h) for h in ("H3", "H4", "H6")]
names = [f"task_meta_{stage}.csv", f"clean_metrics_{stage}.csv", f"event_rows_{stage}.csv.gz"]
if stage == "test":
    names += ["clean_per_forecast.csv.gz", "info_inputs.csv.gz"]
for n in names:
    frames = [pd.read_csv(os.path.join(p, n)) for p in parts]
    out = pd.concat(frames, ignore_index=True)
    kw = dict(compression="gzip") if n.endswith(".gz") else {}
    out.to_csv(os.path.join(D, n), index=False, **kw)
    print(f"{n}: {' + '.join(str(len(f)) for f in frames)} = {len(out)} rows")
