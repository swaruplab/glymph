#!/usr/bin/env python
"""Where do the cross-species counts that a tool leaves behind sit? (species mixtures)

In a barcode of one species every molecule of the other species is background, so whatever a tool
leaves of them is a miss. For each tool the misses are split by
  the size of the entry in the raw matrix (1, 2-3, 4-9, 10 or more counts),
  the cell: how much of its cross-species counts it lost (cells that lost under half, under 90%),
  Glymph's own flags (doublet-like cells are left alone on purpose),
  the genes holding the most of them.

    python diag_unremoved.py --raw data/std/LIB.h5ad --per-cell out/metrics/per_cell/LIB.TOOL.tsv.gz \
        --tool NAME=out/TOOL/LIB.h5ad [NAME=...] [--artifacts out/metrics/artifacts/LIB.tsv]
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
ap.add_argument("--per-cell", required=True, help="per-cell table of any tool: gives the scored cells and their species")
ap.add_argument("--tool", nargs="+", required=True)
ap.add_argument("--artifacts")
ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()
raw = read_counts(a.raw)
pc = pd.read_csv(a.per_cell, sep="\t").set_index("barcode")
spg = gene_species(raw.var)
gname = raw.var["gene_name"].astype(str).to_numpy() if "gene_name" in raw.var else raw.var_names.astype(str).to_numpy()
keep = np.ones(raw.n_vars, bool)
if a.artifacts and os.path.exists(a.artifacts):
    art = pd.read_csv(a.artifacts, sep="\t")
    keep &= ~np.isin(gname, art.gene.astype(str))
ri = raw.obs_names.get_indexer(pc.index)
X = sp.csr_matrix(raw.X)[ri].astype(np.float64)
cell_h = (pc.species == "human").to_numpy()
# cross-species mask per entry: gene of the other species than the cell
Xc = X.tocoo()
cross = np.where(cell_h[Xc.row], spg[Xc.col] == "mouse", spg[Xc.col] == "human") & keep[Xc.col]
Rc = sp.csr_matrix((Xc.data[cross], (Xc.row[cross], Xc.col[cross])), shape=X.shape)
tot = Rc.sum()
print(f"{os.path.basename(a.raw)}: {len(pc)} scored cells, {int(tot)} cross-species counts ({100 * tot / X.sum():.2f}% of their counts) in {Rc.nnz} entries")
bins = [(1, 1), (2, 3), (4, 9), (10, 10 ** 9)]
print("raw entries by size: " + ", ".join(f"{lo}{'' if lo == hi else '+' if hi > 10 ** 6 else '-' + str(hi)}: {100 * Rc.data[(Rc.data >= lo) & (Rc.data <= hi)].sum() / tot:.1f}%" for lo, hi in bins))
for t in a.tool:
    name, path = t.split("=", 1)
    if not os.path.exists(path):
        print(f"{name}: missing"); continue
    o = read_counts(path, integer=False)
    gi = pd.Index(raw.var["gene_id"].astype(str)).get_indexer(o.var["gene_id"].astype(str))
    ok = gi >= 0
    proj = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), gi[ok])), shape=(o.n_vars, raw.n_vars))
    oi = o.obs_names.get_indexer(pc.index)
    if (oi < 0).any():
        print(f"{name}: {int((oi < 0).sum())} scored cells missing from the output"); continue
    Y = (sp.csr_matrix(o.X).astype(np.float64) @ proj).tocsr()[oi]
    # what is left of each raw cross entry (looked up by position; a sparse sum would drop the entries that went to zero)
    Rc.sort_indices()
    Yc = Y.multiply(Rc > 0).tocsr()
    Yc.sort_indices()
    G = Rc.shape[1]
    kr = np.repeat(np.arange(Rc.shape[0], dtype=np.int64), np.diff(Rc.indptr)) * G + Rc.indices
    ky = np.repeat(np.arange(Yc.shape[0], dtype=np.int64), np.diff(Yc.indptr)) * G + Yc.indices
    lv = np.zeros(Rc.nnz)
    lv[np.searchsorted(kr, ky)] = Yc.data
    left = sp.csr_matrix((lv, Rc.indices.copy(), Rc.indptr.copy()), shape=Rc.shape)
    l, r = left.data, Rc.data
    L = l.sum()
    print(f"\n{name}: left {int(round(L))} of {int(tot)} ({100 * L / tot:.2f}%)")
    print("  by raw entry size, share of what is left (share removed within the bin): " + ", ".join(
        f"{lo}{'' if lo == hi else '+' if hi > 10 ** 6 else '-' + str(hi)}: {100 * l[(r >= lo) & (r <= hi)].sum() / max(L, 1):.1f}% ({100 * (1 - l[(r >= lo) & (r <= hi)].sum() / max(r[(r >= lo) & (r <= hi)].sum(), 1)):.1f}%)" for lo, hi in bins))
    cb, ca = np.asarray(Rc.sum(1)).ravel(), np.asarray(left.sum(1)).ravel()
    rm = 1 - ca / np.maximum(cb, 1)
    has = cb >= 5
    for thr in (0.5, 0.9):
        m = has & (rm < thr)
        print(f"  cells that lost under {int(100 * thr)}% of their cross-species counts: {int(m.sum())} of {int(has.sum())}, holding {100 * ca[m].sum() / max(L, 1):.1f}% of what is left ({100 * cb[m].sum() / tot:.1f}% of the raw cross-species counts)")
    q = np.quantile(rm[has], [0.05, 0.25, 0.5, 0.75, 0.95])
    print("  per-cell share removed, 5/25/50/75/95%: " + " ".join(f"{v:.3f}" for v in q))
    # by how dirty the cell is
    fr = cb / np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    order = np.argsort(fr)
    for nm, sl in (("cleanest third", order[: len(order) // 3]), ("middle third", order[len(order) // 3: 2 * len(order) // 3]), ("dirtiest third", order[2 * len(order) // 3:])):
        print(f"  {nm} of cells (cross share {np.median(fr[sl]):.4f}): removed {100 * (1 - ca[sl].sum() / max(cb[sl].sum(), 1)):.1f}%, holds {100 * ca[sl].sum() / max(L, 1):.1f}% of what is left")
    for col in ("glymph_doublet_like",):
        if col in o.obs:
            f = o.obs[col].to_numpy()[oi].astype(bool)
            print(f"  {col}: {int(f.sum())} cells, hold {100 * ca[f].sum() / max(L, 1):.1f}% of what is left ({100 * cb[f].sum() / tot:.1f}% of raw); removal among the others {100 * (1 - ca[~f].sum() / max(cb[~f].sum(), 1)):.2f}%")
    for sp_name, sel in (("human cells", cell_h), ("mouse cells", ~cell_h)):
        if sel.any():
            print(f"  {sp_name}: {int(sel.sum())}, removed {100 * (1 - ca[sel].sum() / max(cb[sel].sum(), 1)):.1f}%")
    gl, gr = np.asarray(left.sum(0)).ravel(), np.asarray(Rc.sum(0)).ravel()
    top = np.argsort(-gl)[: a.top]
    print(f"  genes: {int((gl > 0.5).sum())} hold something; top {a.top} hold {100 * gl[top].sum() / max(L, 1):.1f}%: " + ", ".join(f"{gname[g]}({int(round(gl[g]))}/{int(gr[g])})" for g in top))
    # genes by how much of them was removed
    big = gr >= 50
    grm = 1 - gl[big] / gr[big]
    print(f"  genes with 50 or more cross-species counts: {int(big.sum())}; removed under 50% in {int((grm < 0.5).sum())} (holding {100 * gl[big][grm < 0.5].sum() / max(L, 1):.1f}% of what is left), under 90% in {int((grm < 0.9).sum())} ({100 * gl[big][grm < 0.9].sum() / max(L, 1):.1f}%)")
