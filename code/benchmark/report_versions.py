#!/usr/bin/env python3
"""Head-to-head summary of one all-tier run from its tables (MacBook).

    python3 code/benchmark/report_versions.py runs/20261117-1342_glymph-cb040-bench [tool ...]

Default tools: glymph glymph_rep2 glymph_031 cellbender cellbender032 (as named by relabel_tools.py).
Prints means over the libraries never used in development (all of them, and by group), over all
mixtures by group, the kidney libraries, the number of libraries on which the first tool beats each
other tool, run times, and the spread between the first two tools when they are two runs of one code.
"""
import os
import sys

import pandas as pd

run = sys.argv[1]
tools = sys.argv[2:] or ["glymph", "glymph_rep2", "glymph_031", "cellbender", "cellbender032"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
sp = pd.read_csv(os.path.join(run, "scores_species.tsv"), sep="\t")
kid = pd.read_csv(os.path.join(run, "scores_kidney.tsv"), sep="\t")
res = pd.read_csv(os.path.join(run, "resources.tsv"), sep="\t")
tools = [t for t in tools if (sp.tool == t).any()]
cols = ["cross_species_removed", "own_retained", "same_species_ratio", "native_loss", "contam_rmsle", "contam_kendall"]
grp = {"t1_holdout": "10x cell lines", "t1_tuning": "10x cell lines (tuning)", "t1_nuclei": "nuclei mixture", "t2_xeno": "xenograft", "t3_rhapsody": "BD Rhapsody", "t4_parse": "Parse"}
sp["group"] = sp.tier.map(grp)


def block(t, title):
    have = t.groupby("dataset").tool.apply(set)
    full = [d for d, s in have.items() if set(tools) <= s]
    t = t[t.dataset.isin(full) & t.tool.isin(tools)]
    print(f"\n== {title}: {len(full)} libraries")
    print(t.groupby("tool")[cols].mean().loc[tools].T.to_string())


held = sp[sp.development == "held out"]
block(held, "never used in development, all")
for g in ("10x cell lines", "xenograft", "BD Rhapsody"):
    block(held[held.group == g], f"never used in development, {g}")
for g in ("10x cell lines", "10x cell lines (tuning)", "xenograft", "BD Rhapsody", "Parse", "nuclei mixture"):
    block(sp[sp.group == g], f"all mixtures, {g}")
print(f"\n== kidney: {kid.dataset.nunique()} libraries")
print(kid[kid.tool.isin(tools)].groupby("tool")[["contam_rmsle", "contam_kendall", "removed_over_truth", "marker_removed_nonpt", "marker_kept_pt"]].mean().loc[[t for t in tools if (kid.tool == t).any()]].T.to_string())

both = pd.concat([sp[["dataset", "tool", "contam_rmsle", "contam_kendall"]], kid[["dataset", "tool", "contam_rmsle", "contam_kendall"]]])
e = both.pivot(index="dataset", columns="tool", values="contam_rmsle")
k = both.pivot(index="dataset", columns="tool", values="contam_kendall")
a = tools[0]
print(f"\n== libraries with truth ({len(e)}) on which {a} is better than each other tool")
for b in tools[1:]:
    print(f"   {b:16s} lower per-cell error on {int((e[a] < e[b]).sum())}, higher rank agreement on {int((k[a] > k[b]).sum())}")
    worse = e.index[e[a] >= e[b]]
    if len(worse) and len(worse) <= 8:
        print("      not lower on: " + ", ".join(f"{d} ({e.loc[d, a]:.3f} vs {e.loc[d, b]:.3f})" for d in worse))
print("\n== wall time per library, s")
r = res[res.tool.isin(tools)].groupby("tool").agg(n=("wall_s", "size"), median=("wall_s", "median"), longest=("wall_s", "max"), total_h=("wall_s", lambda v: v.sum() / 3600),
                                                 median_ram_gb=("peak_rss_gb", "median"), peak_gpu_gb=("gpu_peak_mem_gb", "max"))
print(r.loc[[t for t in tools if t in r.index]].to_string())
if len(tools) > 1:
    w = res[res.tool.isin(tools)].pivot(index="dataset", columns="tool", values="wall_s").dropna()
    for b in tools[1:]:
        if b in w:
            print(f"   median of {b} / {a} across libraries: {(w[b] / w[a]).median():.1f}")
if len(tools) > 1 and tools[1].startswith(a):
    print(f"\n== two runs of the same code ({a}, {tools[1]}): absolute difference per library")
    for c in ("cross_species_removed", "own_retained", "contam_rmsle"):
        p = sp[sp.tool.isin(tools[:2])].pivot(index="dataset", columns="tool", values=c).dropna()
        d = (p[a] - p[tools[1]]).abs()
        top = d.sort_values(ascending=False).head(4)
        print(f"   {c:22s} median {d.median():.4f}, mean {d.mean():.4f}, largest: " + ", ".join(f"{i} {v:.3f}" for i, v in top.items()))
    p = sp[sp.tool.isin(tools[:2]) & (sp.development == "held out")].pivot(index="dataset", columns="tool", values="cross_species_removed")
    print("   never-used libraries, cross-species removal, both runs:\n" + p.to_string())
