#!/usr/bin/env python
"""Cross-species counts that the ambient pool cannot explain, gene by gene (species mixtures).

In cells of one species the other species' genes should follow the ambient profile: gene g should
hold the share of the cross-species counts that it holds among the other species' genes in the empty
partitions (or in the other species' cells, whichever is larger). A gene far above that is not
background: reads of the cell's own gene aligned to the other genome (conserved genes, ribosomal and
mitochondrial RNA, processed pseudogenes). The table gives, per fold of excess, how many cross-species
counts sit there and how much of them each tool removes; and the genes holding the most.

    python diag_cross_excess.py --raw data/std/LIB.h5ad --cells out/prep/LIB.cells.txt \
        --per-cell out/metrics/per_cell/LIB.TOOL.tsv.gz --tool NAME=path [NAME=path ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species, read_counts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--per-cell", required=True)
ap.add_argument("--tool", nargs="+", required=True)
ap.add_argument("--top", type=int, default=14)
a = ap.parse_args()
raw = read_counts(a.raw)
X = sp.csr_matrix(raw.X).astype(np.float64)
N = np.asarray(X.sum(1)).ravel()
listed = raw.obs_names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(N[listed], 0.01))
bg = (N >= 5) & (N <= T / 10.0)
soup = np.asarray(X[bg].sum(0)).ravel()
pc = pd.read_csv(a.per_cell, sep="\t").set_index("barcode")
ri = raw.obs_names.get_indexer(pc.index)
Xe = X[ri]
spg = gene_species(raw.var)
gname = raw.var["gene_name"].astype(str).to_numpy() if "gene_name" in raw.var else raw.var_names.astype(str).to_numpy()
cell_h = (pc.species == "human").to_numpy()
outs = {}
for t in a.tool:
    name, path = t.split("=", 1)
    if not os.path.exists(path):
        print(f"{name}: missing ({path})"); continue
    o = read_counts(path, integer=False)
    gi = pd.Index(raw.var["gene_id"].astype(str)).get_indexer(o.var["gene_id"].astype(str))
    ok = gi >= 0
    proj = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), gi[ok])), shape=(o.n_vars, raw.n_vars))
    oi = o.obs_names.get_indexer(pc.index)
    have = oi >= 0
    outs[name] = ((sp.csr_matrix(o.X).astype(np.float64) @ proj).tocsr(), oi, have)
print(f"{os.path.basename(a.raw)}: {len(pc)} scored cells, {int(bg.sum())} empty partitions")
edges = [0, 1.5, 3, 10, np.inf]
tot_rows = {}
for o_sp, sel, other in (("human", cell_h, "mouse"), ("mouse", ~cell_h, "human")):
    if sel.sum() < 20 or (~sel).sum() < 20:
        continue
    og = spg == other
    obs = np.asarray(Xe[sel].sum(0)).ravel() * og
    src = np.asarray(Xe[~sel].sum(0)).ravel() * og
    sp_ = soup * og
    ref = np.maximum(sp_ / max(sp_.sum(), 1), src / max(src.sum(), 1))
    exp = ref * obs.sum()
    fold = obs / np.maximum(exp, 0.5)
    left = {}
    for name, (Y, oi, have) in outs.items():
        rows = np.flatnonzero(sel & have)
        left[name] = (np.asarray(Y[oi[rows]].sum(0)).ravel() * og, np.asarray(Xe[rows].sum(0)).ravel() * og)
    print(f"\n{other} genes in {o_sp} cells ({int(sel.sum())} cells, {int(obs.sum())} counts)")
    print("  excess over the ambient profile\tgenes\tshare of cross-species counts\t" + "\t".join(f"removed by {n}" for n in left))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fold >= lo) & (fold < hi) & (obs >= 20)
        key = f"{lo}-{hi}"
        row = [f"  {lo:g} to {hi:g}x", str(int(m.sum())), f"{100 * obs[m].sum() / obs.sum():.1f}%"]
        for n, (l, r) in left.items():
            row.append(f"{100 * (1 - l[m].sum() / max(r[m].sum(), 1)):.1f}%")
            a_, b_ = tot_rows.setdefault((key, n), [0.0, 0.0])
            tot_rows[(key, n)] = [a_ + l[m].sum(), b_ + r[m].sum()]
        print("\t".join(row))
    m = obs < 20
    print(f"  genes under 20 counts\t{int((m & (obs > 0)).sum())}\t{100 * obs[m].sum() / obs.sum():.1f}%\t" + "\t".join(f"{100 * (1 - l[m].sum() / max(r[m].sum(), 1)):.1f}%" for l, r in left.values()))
    for n, (l, r) in left.items():
        a_, b_ = tot_rows.setdefault(("small", n), [0.0, 0.0])
        tot_rows[("small", n)] = [a_ + l[m].sum(), b_ + r[m].sum()]
    top = np.argsort(-(obs - exp) * (fold >= 3))[: a.top]
    print("  genes with the most counts beyond the ambient profile (observed, expected; kept by each tool):")
    for g in top:
        if fold[g] < 3:
            break
        print(f"    {gname[g]}\t{int(obs[g])}\t{exp[g]:.0f}\t" + "\t".join(f"{n} {100 * l[g] / max(r[g], 1):.0f}%" for n, (l, r) in left.items()))
print("\nboth directions together: share removed, by excess over the ambient profile")
names = list(outs)
for key in [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])] + ["small"]:
    print(f"  {key}\t" + "\t".join(f"{n} {100 * (1 - tot_rows[(key, n)][0] / max(tot_rows[(key, n)][1], 1)):.1f}%" for n in names if (key, n) in tot_rows))
for n in names:
    lo_l = sum(v[0] for (k, nn), v in tot_rows.items() if nn == n and k in ("0-1.5", "1.5-3", "small"))
    lo_r = sum(v[1] for (k, nn), v in tot_rows.items() if nn == n and k in ("0-1.5", "1.5-3", "small"))
    al_l = sum(v[0] for (k, nn), v in tot_rows.items() if nn == n)
    al_r = sum(v[1] for (k, nn), v in tot_rows.items() if nn == n)
    print(f"  {n}: removed {100 * (1 - al_l / max(al_r, 1)):.2f}% of all cross-species counts, {100 * (1 - lo_l / max(lo_r, 1)):.2f}% of those in genes within 3x of the ambient profile ({100 * lo_r / max(al_r, 1):.1f}% of the counts)")
