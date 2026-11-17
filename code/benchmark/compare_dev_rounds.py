#!/usr/bin/env python3
"""Compact comparison of development variants (MacBook; reads per-library metric tables).

    python3 code/benchmark/compare_dev_rounds.py <metrics dir> tool1 tool2 ... [--pairs a:b ...]

Prints the mean of each score over the species mixtures and over the kidney libraries that every
listed tool has, the number of libraries on which each tool beats the first one listed, and for
each replicate pair a:b the mean and the largest absolute difference between the two runs.
"""
import glob
import os
import sys

import pandas as pd

args = sys.argv[1:]
pairs = []
if "--pairs" in args:
    i = args.index("--pairs")
    pairs = [p.split(":") for p in args[i + 1:]]
    args = args[:i]
mdir, tools = args[0], args[1:]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
m = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(mdir, "*.tsv"))], ignore_index=True)
have = m.groupby("dataset").tool.apply(set)
full = [d for d, t in have.items() if set(tools) <= t]
m = m[m.dataset.isin(full)]
BETTER_HIGH = {"cross_species_removed", "own_retained", "contam_kendall", "marker_removed_nonpt", "marker_kept_pt"}


def block(t, cols, title):
    if not len(t):
        return
    print(f"== {title}: {t.dataset.nunique()} libraries")
    rows = {}
    for c in cols:
        p = t[t.tool.isin(tools)].pivot(index="dataset", columns="tool", values=c)[tools]
        rows[c] = p.mean()
        if c in BETTER_HIGH or c in ("contam_rmsle",):
            ref = p[tools[0]]
            wins = {x: int(((p[x] > ref) if c in BETTER_HIGH else (p[x] < ref)).sum()) for x in tools[1:]}
            rows[c + "  (libraries better than " + tools[0] + ")"] = pd.Series(wins)
    print(pd.DataFrame(rows).T[tools].to_string())
    for a, b in pairs:
        if a in t.tool.values and b in t.tool.values:
            d = {}
            for c in cols:
                p = t[t.tool.isin([a, b])].pivot(index="dataset", columns="tool", values=c)
                diff = (p[a] - p[b]).abs()
                d[c] = f"mean |diff| {diff.mean():.4f}, max {diff.max():.4f} ({diff.idxmax()})"
            print(f"  replicates {a} vs {b}:")
            for c, v in d.items():
                print(f"    {c:24s} {v}")


sp = m[m.get("cross_species_removed").notna()] if "cross_species_removed" in m else m.iloc[:0]
block(sp, ["cross_species_removed", "own_retained", "same_species_ratio", "native_loss", "contam_rmsle", "contam_kendall"], "species mixtures")
kid = m[m.dataset.str.startswith("kidney")]
block(kid, ["contam_rmsle", "contam_kendall", "removed_over_truth", "marker_removed_nonpt", "marker_kept_pt"], "kidney")
if "--per-library" in sys.argv or os.environ.get("PER_LIBRARY"):
    for c in ("cross_species_removed", "contam_rmsle"):
        print(sp[sp.tool.isin(tools)].pivot(index="dataset", columns="tool", values=c)[tools].to_string())
