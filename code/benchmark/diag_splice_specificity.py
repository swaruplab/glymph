#!/usr/bin/env python
"""How unspliced are a gene's molecules in cell types that express it less and less? (kidney nuclei, authors' cell types)

For every gene and cell type: the gene's rate there over its rate in the cell type that expresses it most. Pairs are
pooled by that ratio. If the molecules in the least-expressing cell types are background, their unspliced share
falls to the background's own value as the ratio goes to zero; that value, against the empty droplets', is the
shift between background outside and inside cells.

    python diag_splice_specificity.py --replicate nuc2 --raw data/std/kidney_nuc2.h5ad --splice data/splice/kidney_nuc2 \
        --meta data/kidney_strainmix/cell_metadata.tsv.gz --cells out/prep/kidney_nuc2.cells.txt
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
for k in ("replicate", "raw", "splice", "meta", "cells"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
raw = ad.read_h5ad(a.raw, backed="r")
names = pd.Index(raw.obs_names)
U, A = (sp.load_npz(f"{a.splice}.{k}.npz").tocsr().astype(np.float64) for k in ("unspliced", "ambiguous"))
M = (U + A).tocsr()
tot = np.asarray(M.sum(1)).ravel()
cells = names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(tot[cells], 0.01))
empty = (tot >= 5) & (tot <= T / 10.0)
meta = pd.read_csv(a.meta, sep="\t")
meta = meta[meta.replicate == a.replicate].set_index("cell")
ct = meta.celltype.reindex(names)
types = [c for c, n in ct.value_counts().items() if n >= 100]
Xc = np.stack([np.asarray(M[(ct == c).to_numpy()].sum(0)).ravel() for c in types])
Uc = np.stack([np.asarray(U[(ct == c).to_numpy()].sum(0)).ravel() for c in types])
Nc = Xc.sum(1, keepdims=True)
rate = Xc / Nc
top = rate.argmax(0)
g = np.arange(Xc.shape[1])
ratio = rate / np.maximum(rate[top, g], 1e-30)
xe, ue = np.asarray(M[empty].sum(0)).ravel(), np.asarray(U[empty].sum(0)).ravel()
pe = ue.sum() / xe.sum()
pi_amb = (ue + 10 * pe) / (xe + 10)
pi_top = Uc[top, g] / np.maximum(Xc[top, g], 1)
good = (Xc[top, g] >= 50) & (xe >= 10) & (pi_top - pi_amb >= 0.1)
print(f"{a.replicate}: cell types {types}; genes used {int(good.sum())} (50 molecules or more in their top cell type, 10 or more in the empty droplets, gap of 0.1 or more)")
print("rate over the top cell type's\tpairs\tmolecules\tunspliced share\tif they were the empty droplets' RNA\tif they were own expression\timplied share of own expression")
edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ratio >= lo) & (ratio < hi) & good[None, :] & (Xc >= 5)
    if hi > 1:
        m &= True
    x, u = Xc[m].sum(), Uc[m].sum()
    amb = (Xc * pi_amb[None, :])[m].sum() / max(x, 1)
    nat = (Xc * pi_top[None, :])[m].sum() / max(x, 1)
    obs = u / max(x, 1)
    print(f"{lo:.2f} to {min(hi, 1):.2f}\t{int(m.sum())}\t{int(x)}\t{obs:.3f}\t{amb:.3f}\t{nat:.3f}\t{(obs - amb) / max(nat - amb, 1e-9):.2f}")
