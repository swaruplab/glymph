#!/usr/bin/env python
"""Can splicing alone measure how much of a cell type's RNA is background? (kidney nuclei, genotype truth)

For gene g in cell type c, with pi_amb[g] from the empty droplets and pi_nat[g] the unspliced share of the gene's own
expression, the background share of its molecules there is f = (pi_nat - observed share) / (pi_nat - pi_amb).
pi_nat comes from the gene's top cell type, corrected for the background that cell type carries, taken here as the
gene's lowest rate in any cell type over its top rate (no model, no truth). Summed over genes this gives a background
fraction per cell type, printed beside the genotype truth of the same cells and the fraction removed by any tool.

    python diag_splice_amount.py --replicate nuc2 --raw data/std/kidney_nuc2.h5ad --splice data/splice/kidney_nuc2 \
        --meta data/kidney_strainmix/cell_metadata.tsv.gz --cells out/prep/kidney_nuc2.cells.txt [--per-cell NAME=table ...]
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
for k in ("replicate", "raw", "splice", "meta", "cells"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--per-cell", nargs="*", default=[])
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
meta = meta[(meta.replicate == a.replicate) & meta.contPerCell_binom.notna()].set_index("cell")
ct = meta.celltype.reindex(names)
truth = meta.contPerCell_binom.reindex(names).to_numpy(float)
types = [c for c, n in ct.value_counts().items() if n >= 100]
sel = {c: (ct == c).to_numpy() for c in types}
Xc = np.stack([np.asarray(M[sel[c]].sum(0)).ravel() for c in types])
Uc = np.stack([np.asarray(U[sel[c]].sum(0)).ravel() for c in types])
rate = Xc / Xc.sum(1, keepdims=True)
g = np.arange(Xc.shape[1])
top = rate.argmax(0)
low = np.where(Xc >= 0, rate, np.inf).min(0)
r = np.clip(low / np.maximum(rate[top, g], 1e-30), 0, 0.9)  # background share of the gene's molecules in its top cell type
xe, ue = np.asarray(M[empty].sum(0)).ravel(), np.asarray(U[empty].sum(0)).ravel()
pe = ue.sum() / xe.sum()
pi_amb = (ue + 10 * pe) / (xe + 10)
pi_top = Uc[top, g] / np.maximum(Xc[top, g], 1)
pi_nat = np.clip((pi_top - r * pi_amb) / (1 - r), 0, 0.999)
good = (Xc[top, g] >= 50) & (pi_nat - pi_amb >= 0.15)
print(f"{a.replicate}: {int(good.sum())} informative genes holding {100 * Xc[:, good].sum() / Xc.sum():.0f}% of the molecules; pi_amb median {np.median(pi_amb[good]):.3f}, pi_nat median {np.median(pi_nat[good]):.3f}")
est = {}
for nm_path in a.per_cell:
    nm, path = nm_path.split("=", 1)
    t = pd.read_csv(path, sep="\t").set_index("barcode")
    est[nm] = t.est_frac.reindex(names).to_numpy(float)
print("cell type\tcells\ttruth (count-weighted)\tfrom splicing, informative genes\t" + "\t".join(est))
out = []
for i, c in enumerate(types):
    obs = Uc[i] / np.maximum(Xc[i], 1)
    f = np.clip((pi_nat - obs) / np.maximum(pi_nat - pi_amb, 1e-9), 0, 1)
    F = (f * Xc[i])[good].sum() / Xc[i][good].sum()
    w = tot[sel[c]]
    tr = np.nansum(truth[sel[c]] * w) / w.sum()
    row = [c, int(sel[c].sum()), tr, F] + [np.nansum(np.nan_to_num(e[sel[c]]) * w * ~np.isnan(e[sel[c]])) / max((w * ~np.isnan(e[sel[c]])).sum(), 1) for e in est.values()]
    out.append(row)
    print("\t".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))
o = np.array([[r_[2], r_[3]] + r_[4:] for r_ in out], float)
w = np.array([r_[1] for r_ in out], float)
print("over the cell types (weighted by cells): truth %.3f, splicing %.3f" % (np.average(o[:, 0], weights=w), np.average(o[:, 1], weights=w)) + "".join(f", {n} {np.average(o[:, 2 + j], weights=w):.3f}" for j, n in enumerate(est)))
if len(types) >= 4:
    print("correlation with the truth across cell types: splicing %.2f" % np.corrcoef(o[:, 0], o[:, 1])[0, 1] + "".join(f", {n} {np.corrcoef(o[:, 0], o[:, 2 + j])[0, 1]:.2f}" for j, n in enumerate(est)))
