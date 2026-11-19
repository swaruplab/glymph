#!/usr/bin/env python3
"""Rename tools in the tables of a run folder, so that figures and pages read each tool by its role.

    python3 code/benchmark/relabel_tools.py runs/<run> old=new [old=new ...]

Every .tsv directly in the run folder that has a "tool" column is rewritten; the tables as they
were scored are kept under as_scored/ the first time, and the renaming is written to tool_names.tsv.
Renames are applied at once, so two names can be swapped. Example (the CellBender 0.4.0 benchmark):
the scored folder cellbender040 becomes "cellbender" and the 0.3.2 results become "cellbender032".
"""
import glob
import os
import shutil
import sys

import pandas as pd

run = sys.argv[1]
ren = dict(a.split("=", 1) for a in sys.argv[2:])
keep = os.path.join(run, "as_scored")
first = not os.path.isdir(keep)
os.makedirs(keep, exist_ok=True)
n = 0
for f in sorted(glob.glob(os.path.join(run, "*.tsv"))):
    base = os.path.basename(f)
    if base in ("jobs.tsv", "decisions.tsv", "tool_names.tsv"):
        continue
    src = os.path.join(keep, base)
    if first or not os.path.exists(src):
        shutil.copy2(f, src)
    t = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)  # always from the scored copy, so a rerun gives the same tables
    if "tool" not in t.columns:
        continue
    t["tool"] = t["tool"].map(lambda x: ren.get(x, x))
    t.to_csv(f, sep="\t", index=False)
    n += 1
pd.DataFrame({"scored_as": list(ren), "named": list(ren.values())}).to_csv(os.path.join(run, "tool_names.tsv"), sep="\t", index=False)
print(f"{n} tables renamed in {run}; originals in {keep}")
