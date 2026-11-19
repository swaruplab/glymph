#!/usr/bin/env python
"""What is the unspliced share of the background that actually sits in nuclei? (kidney, authors' cell types)

Proximal-tubule marker genes are background in every other cell type. Their molecules there are compared, by
unspliced share, with the same genes in the empty droplets and in proximal-tubule nuclei. If the background in
nuclei is the empty droplets' RNA, the first two agree.

    python diag_splice_markers.py --replicate nuc2 --raw data/std/kidney_nuc2.h5ad --splice data/splice/kidney_nuc2 \
        --meta data/kidney_strainmix/cell_metadata.tsv.gz --markers data/kidney_strainmix/top10_PT_markers.txt --cells out/prep/kidney_nuc2.cells.txt
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
for k in ("replicate", "raw", "splice", "meta", "markers", "cells"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
raw = ad.read_h5ad(a.raw, backed="r")
names = pd.Index(raw.obs_names)
U, A = (sp.load_npz(f"{a.splice}.{k}.npz").tocsr().astype(np.float64) for k in ("unspliced", "ambiguous"))
M = U + A
tot = np.asarray(M.sum(1)).ravel()
gname = raw.var["gene_name"].astype(str).to_numpy()
mk = [x.strip() for x in open(a.markers) if x.strip()]
gi = np.flatnonzero(np.isin(gname, mk))
cells = names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(tot[cells], 0.01))
empty = (tot >= 5) & (tot <= T / 10.0)
meta = pd.read_csv(a.meta, sep="\t")
meta = meta[meta.replicate == a.replicate].set_index("cell")
ct = meta.celltype.reindex(names).to_numpy()
groups = {"empty droplets": empty, "proximal tubule nuclei": ct == "PT", "all other cell types": pd.notna(ct) & (ct != "PT")}
print(f"{len(gi)} marker genes found: {' '.join(gname[gi])}")
print("group\tbarcodes\tmarker molecules\tunspliced share of marker molecules\tunspliced share of all molecules\tmarker molecules per 1,000")
for name, m in groups.items():
    u, t = U[m][:, gi].sum(), M[m][:, gi].sum()
    print(f"{name}\t{int(m.sum())}\t{int(t)}\t{u / max(t, 1):.3f}\t{U[m].sum() / max(M[m].sum(), 1):.3f}\t{1000 * t / max(M[m].sum(), 1):.1f}")
for c in pd.Series(ct[pd.notna(ct) & (ct != "PT")]).value_counts().index[:6]:
    m = ct == c
    u, t = U[m][:, gi].sum(), M[m][:, gi].sum()
    print(f"   {c}\t{int(m.sum())}\t{int(t)}\t{u / max(t, 1):.3f}\t{U[m].sum() / max(M[m].sum(), 1):.3f}\t{1000 * t / max(M[m].sum(), 1):.1f}")
print("per marker gene: unspliced share in empty droplets / other cell types / proximal tubule")
oth = groups["all other cell types"]
for g in gi:
    r = [U[m][:, g].sum() / max(M[m][:, g].sum(), 1) for m in (empty, oth, ct == "PT")]
    print(f"   {gname[g]:12s} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}   molecules {int(M[empty][:, g].sum())} / {int(M[oth][:, g].sum())} / {int(M[ct == 'PT'][:, g].sum())}")
