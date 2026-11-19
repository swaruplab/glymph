#!/usr/bin/env python3
"""Tool by library tables of the scores that need no truth (MacBook; reads out/generic tables).

    python3 code/benchmark/compare_generic.py <generic dir> tool1 tool2 ... [--only lib1,lib2]

removed_pct          share of the called cells' counts removed
marker_kept_on       share of a group's own exclusive-marker counts kept (higher is better)
marker_removed_off   share of exclusive-marker counts removed from the groups that do not express them (higher is better)
"""
import glob
import os
import sys

import pandas as pd

args = sys.argv[1:]
only = None
if "--only" in args:
    i = args.index("--only")
    only = args[i + 1].split(",")
    args = args[:i]
gdir, tools = args[0], args[1:]
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
g = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(gdir, "*.tsv"))], ignore_index=True)
g = g[g.tool.isin(tools)]
if only:
    g = g[g.dataset.isin(only)]
for c in ("removed_pct", "marker_kept_on", "marker_removed_off"):
    p = g.pivot(index="dataset", columns="tool", values=c).reindex(columns=tools)
    full = p.dropna()
    print(f"\n{c}")
    print(p.to_string())
    if len(full):
        print(f"mean over the {len(full)} libraries every tool has".ljust(24) + "  " + "  ".join(f"{t} {full[t].mean():.4g}" for t in tools))
bad = g[(g.n_above_raw > 0) | (g.n_negative > 0) | (g.n_noninteger > 0)]
print("\noutputs that add counts, go negative or are not integers:", "none" if not len(bad) else bad[["dataset", "tool", "n_above_raw", "n_negative", "n_noninteger"]].to_string(index=False))
