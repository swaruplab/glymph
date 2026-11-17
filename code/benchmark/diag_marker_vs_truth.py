#!/usr/bin/env python
"""Kidney: do proximal tubule (PT) marker counts in non-PT cells match the genotype-based truth?

The genotype truth sees background that came from another mouse. PT-marker counts in a non-PT cell
are background whatever their origin. If the ambient profile were all there is, the marker counts
would equal truth background x the markers' share of the soup. A ratio well above 1 means non-PT
cells carry PT RNA that the genotype truth does not count (same-mouse material, doublets).

    python diag_marker_vs_truth.py <glymph .h5ad with layers['raw']> <replicate> <cell_metadata.tsv.gz> <markers.txt>
"""
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

a = ad.read_h5ad(sys.argv[1])
rep = sys.argv[2]
meta = pd.read_csv(sys.argv[3], sep="\t")
meta = meta[(meta.replicate == rep) & meta.contPerCell_binom.notna()].drop_duplicates("cell").set_index("cell")
want = [g.strip() for g in open(sys.argv[4]) if g.strip()]
X = sp.csr_matrix(a.layers["raw"]).astype(np.float64)
gname = a.var["gene_name"].astype(str).to_numpy()
mk = np.isin(gname, want)
soup = a.var["glymph_soup_mean"].to_numpy(float)
soup = soup / soup.sum()
share = soup[mk].sum()
idx = a.obs_names.get_indexer(meta.index)
ok = idx >= 0
meta, idx = meta[ok], idx[ok]
N = np.asarray(X[idx].sum(1)).ravel()
y = np.asarray(X[idx][:, mk].sum(1)).ravel()
b = meta.contPerCell_binom.to_numpy(float) * N
print(f"{rep}: {mk.sum()} markers, share of soup {share:.3f}; cells matched {len(idx)}")
for lab, sel in (("non-PT", (meta.celltype != "PT").to_numpy() & meta.celltype.notna().to_numpy()), ("PT", (meta.celltype == "PT").to_numpy())):
    for strain in ("all", "CAST", "BL6", "SvImJ"):
        s_ = sel & ((meta.Strain == strain).to_numpy() if strain != "all" else True)
        if s_.sum() < 30:
            continue
        pred = b[s_] * share
        print(f"   {lab:6s} {strain:5s} n {s_.sum():6d}  median depth {np.median(N[s_]):6.0f}  truth background/cell median {np.median(b[s_]):5.0f}  "
              f"marker counts/cell median {np.median(y[s_]):6.1f} mean {y[s_].mean():7.1f}  predicted from truth: median {np.median(pred):5.1f} mean {pred.mean():6.1f}  "
              f"observed/predicted (means) {y[s_].mean() / pred.mean():5.2f}  (medians) {np.median(y[s_]) / max(np.median(pred), 1e-9):5.2f}")
    if lab == "non-PT":
        q = np.quantile(y[sel], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        print("   non-PT marker counts per cell, quantiles 10/25/50/75/90/99:", " ".join(f"{v:.0f}" for v in q))
