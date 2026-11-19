#!/usr/bin/env python
"""Does the unspliced share of a cell's molecules measure its background? (kidney strain mixture, genotype truth)

Read-only feasibility check for a second, independent measurement of the background of nuclei. Ambient RNA in a
nuclei preparation is mostly cytoplasmic, so it is spliced; a nucleus's own RNA is mostly unspliced. With
    pi_amb[g]   unspliced share of gene g among the empty droplets' molecules (measured, a negative control)
    pi_nat[g]   unspliced share of gene g in the cleanest fifth of the cells (those with the highest unspliced share)
a cell with total N, unspliced U and counts x[g] that carries a background fraction F with the soup profile has
    U = sum_g x[g] pi_nat[g] - F N sum_g soup[g] (pi_nat[g] - pi_amb[g])
so F can be read off U. Printed: how the molecule counts compare with the count matrix, the unspliced share of
empty droplets and of cells, and for the cells with a genotype truth the rank and linear agreement of (a) the raw
unspliced share and (b) the estimate F with the truth, beside the agreement any other per-cell table reaches
(--per-cell: tables of kidney_metrics.py, columns truth_frac and est_frac).

    python diag_splice_truth.py --name kidney_nuc2 --replicate nuc2 --raw data/std/kidney_nuc2.h5ad \
        --splice data/splice/kidney_nuc2 --meta data/kidney_strainmix/cell_metadata.tsv.gz --cells out/prep/kidney_nuc2.cells.txt
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau, pearsonr, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--replicate", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--splice", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--per-cell", nargs="*", default=[], help="NAME=per-cell table of another estimate")
ap.add_argument("--out", help="optional table: barcode, truth, unspliced share, estimate")
a = ap.parse_args()

raw = ad.read_h5ad(a.raw)
X = sp.csr_matrix(raw.X)
U, S, A = (sp.load_npz(f"{a.splice}.{k}.npz").tocsr().astype(np.float64) for k in ("unspliced", "spliced", "ambiguous"))
N = np.asarray(X.sum(1)).ravel()
u, s_, am = (np.asarray(M.sum(1)).ravel() for M in (U, S, A))
tot = u + s_ + am
cells = raw.obs_names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(N[cells], 0.01))
empty = (N >= 5) & (N <= T / 10.0)
big = N >= 50
print(f"== {a.name}: {int(cells.sum())} cells (depth at least {T:.0f}), {int(empty.sum())} empty droplets with 5 to {T / 10:.0f} counts")
print(f"molecules counted from the alignment file over the count matrix: all barcodes {tot.sum() / N.sum():.3f}; cells median {np.median(tot[cells] / N[cells]):.3f}; "
      f"correlation of totals over barcodes with 50 counts or more {np.corrcoef(tot[big], N[big])[0, 1]:.4f}")
share = lambda m: 100 * np.array([u[m].sum(), s_[m].sum(), am[m].sum()]) / max(tot[m].sum(), 1)
print("unspliced / spliced / ambiguous share of molecules, %: empty droplets " + " / ".join(f"{v:.1f}" for v in share(empty)) + "; cells " + " / ".join(f"{v:.1f}" for v in share(cells)))
f = u / np.maximum(tot, 1)
q = np.quantile(f[cells], [0.05, 0.25, 0.5, 0.75, 0.95])
print("unspliced share per cell, quantiles 5 25 50 75 95: " + " ".join(f"{v:.3f}" for v in q) + f"; empty droplets pooled {u[empty].sum() / max(tot[empty].sum(), 1):.3f}")

# per-gene unspliced share: background from the empty droplets, native from the cleanest fifth of cells
ue, te = np.asarray(U[empty].sum(0)).ravel(), np.asarray((U + S + A)[empty].sum(0)).ravel()
clean = cells & (f >= np.quantile(f[cells], 0.8))
un, tn = np.asarray(U[clean].sum(0)).ravel(), np.asarray((U + S + A)[clean].sum(0)).ravel()
a0 = 5.0
pe0, pn0 = ue.sum() / te.sum(), un.sum() / tn.sum()
pi_amb = (ue + a0 * pe0) / (te + a0)
pi_nat = (un + a0 * pn0) / (tn + a0)
soup = np.asarray(X[empty].sum(0)).ravel().astype(float)
soup /= soup.sum()
both = (te >= 30) & (tn >= 30)
print(f"per gene ({int(both.sum())} genes with 30 molecules or more in both): unspliced share in empty droplets median {np.median(ue[both] / te[both]):.3f}, "
      f"in the cleanest fifth of cells {np.median(un[both] / tn[both]):.3f}; correlation across genes {np.corrcoef(ue[both] / te[both], un[both] / tn[both])[0, 1]:.2f}; "
      f"genes where the cells' share is higher {100 * np.mean(un[both] / tn[both] > ue[both] / te[both]):.0f}%")
gap = float(soup @ (pi_nat - pi_amb))
M = U + S + A
exp_nat = np.asarray(M @ pi_nat).ravel()  # unspliced molecules expected if every molecule were the cell's own
F = (exp_nat - u) / np.maximum(tot * gap, 1e-9)
print(f"soup-weighted gap between native and background unspliced share: {gap:.3f}")

meta = pd.read_csv(a.meta, sep="\t")
meta = meta[(meta.replicate == a.replicate) & meta.contPerCell_binom.notna()].set_index("cell")
idx = pd.Index(raw.obs_names).get_indexer(meta.index)
ok = idx >= 0
truth = meta.contPerCell_binom.to_numpy(float)[ok]
ii = idx[ok]
keep = tot[ii] >= 200
truth, ii = truth[keep], ii[keep]


def agree(est, name):
    e = np.clip(est, 0, 1)
    rm = float(np.sqrt(np.mean((np.log1p(e) - np.log1p(truth)) ** 2)))
    print(f"   {name:34s} Kendall {kendalltau(est, truth)[0]:6.3f}  Spearman {spearmanr(est, truth)[0]:6.3f}  Pearson {pearsonr(est, truth)[0]:6.3f}  "
          f"RMSLE {rm:.4f}  mean estimate {100 * np.mean(e):5.1f}% (truth {100 * np.mean(truth):.1f}%)")


print(f"cells with a genotype truth and 200 molecules or more: {truth.size}; truth quantiles 5 50 95: " + " ".join(f"{v:.3f}" for v in np.quantile(truth, [0.05, 0.5, 0.95])))
agree(1 - f[ii], "1 - unspliced share (raw)")
agree(F[ii], "background from the unspliced share")
agree(np.clip(F[ii], 0, 1) * 0 + np.mean(truth), "a constant (for the RMSLE scale)") if False else None
for spec in a.per_cell:
    nm, path = spec.split("=", 1)
    if not os.path.exists(path):
        continue
    t = pd.read_csv(path, sep="\t").set_index("barcode")
    est = t.est_frac.reindex(raw.obs_names[ii]).to_numpy(float)
    m = ~np.isnan(est)
    tr = truth[m]
    rm = float(np.sqrt(np.mean((np.log1p(np.clip(est[m], 0, 1)) - np.log1p(tr)) ** 2)))
    print(f"   {nm:34s} Kendall {kendalltau(est[m], tr)[0]:6.3f}  Spearman {spearmanr(est[m], tr)[0]:6.3f}  Pearson {pearsonr(est[m], tr)[0]:6.3f}  RMSLE {rm:.4f}  "
          f"mean estimate {100 * np.mean(est[m]):5.1f}% ({int(m.sum())} cells)")
    both_ok = m
    comb = 0.5 * (np.clip(F[ii][m], 0, 1) + est[m])
    rm = float(np.sqrt(np.mean((np.log1p(comb) - np.log1p(tr)) ** 2)))
    print(f"   {'mean of it and the splicing estimate':34s} Kendall {kendalltau(comb, tr)[0]:6.3f}  Spearman {spearmanr(comb, tr)[0]:6.3f}  Pearson {pearsonr(comb, tr)[0]:6.3f}  RMSLE {rm:.4f}")
if a.out:
    pd.DataFrame({"barcode": raw.obs_names[ii], "truth": truth, "unspliced_share": f[ii], "splice_estimate": F[ii], "molecules": tot[ii]}).to_csv(a.out, sep="\t", index=False)
