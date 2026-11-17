#!/usr/bin/env python
"""Can the swap fraction be read off the data without species labels?

Idea (an exclusion bound). For a cluster c of similar cells and a gene g, the cluster's count
total X_cg is native + ambient + swapped, and native is never negative, so

    u_cg = X_cg / (N_c * chibar_g)  -  (e * n_c / N_c) * soup_g / chibar_g   >=   s

with near-equality for genes the cluster does not express. Expressed genes give u around 1;
silent genes pile up at s. This script clusters Glymph's latent space with k-means, computes
u for every (cluster, gene) with enough expected counts, and reports the lowest peak of the
histogram of log10(u). It never looks at the species of a gene.

    python diag_exclusion.py --run-dir . --tool glymph
"""

import argparse
import glob
import json
import os

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=".")
ap.add_argument("--tool", default="glymph")
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--min-expected", type=float, default=30.0, help="expected swapped counts at s = 1%% for a pair to count")
ap.add_argument("--confident", action="store_true", help="use only barcodes at or above the cell count threshold")
ap.add_argument("--only", default="", help="substring of the library name")
a = ap.parse_args()

print("library            n_cells  k  pairs_used  pairs_below_10pct  s_peak_pct  s_median_below_10pct  glymph_s_pct  e")
for f in sorted(glob.glob(os.path.join(a.run_dir, f"out/{a.tool}/*.h5ad"))):
    name = os.path.basename(f)[: -len(".h5ad")]
    rep = json.load(open(f[: -len(".h5ad")] + ".report.json"))
    e = float(list(rep["e_unit"].values())[0])
    s_model = float(list(rep["swap_fraction"].values())[0])
    if a.only and a.only not in name:
        continue
    A = ad.read_h5ad(f)
    if a.confident:
        A = A[np.asarray(A.layers["raw"].sum(1)).ravel() >= rep["priors"]["cell_threshold"]].copy()
    X = sp.csr_matrix(A.layers["raw"]).astype(np.float64)
    N = np.asarray(X.sum(1)).ravel()
    soup = A.var["glymph_soup_mean"].to_numpy().astype(np.float64)
    soup = soup / soup.sum()
    chibar = np.asarray(X.sum(0)).ravel()
    chibar = chibar / chibar.sum()
    lab = KMeans(n_clusters=a.k, n_init=3, random_state=0).fit_predict(A.obsm["X_glymph"])
    us, ws = [], []
    for c in range(a.k):
        sel = lab == c
        if sel.sum() < 20:
            continue
        Xc = np.asarray(X[sel].sum(0)).ravel()
        Nc, nc = N[sel].sum(), sel.sum()
        ok = (0.01 * Nc * chibar >= a.min_expected) & (chibar > 0)
        u = Xc[ok] / (Nc * chibar[ok]) - (e * nc / Nc) * soup[ok] / chibar[ok]
        us.append(u)
        ws.append(np.full(u.size, 1.0))
    per = []
    for uc in us:
        h, ed = np.histogram(np.log10(np.clip(uc, 1e-4, 1e2)), bins=np.arange(-4, 2.01, 0.1))
        h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
        k_ = int(np.argmax(np.where(ed[:-1] < -1.0, h, -1)))
        if ed[k_ + 1] < -1.0 - 1e-9 and h[k_] >= 5:
            per.append(10 ** (0.5 * (ed[k_] + ed[k_ + 1])))
    u = np.concatenate(us)
    low = u[(u > 1e-4) & (u < 0.1)]
    hist, edges = np.histogram(np.log10(np.clip(u, 1e-4, 1e2)), bins=np.arange(-4, 2.01, 0.1))
    sm = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    lo_bins = edges[:-1] < -1.0  # only look below 10%
    peak = int(np.argmax(np.where(lo_bins, sm, -1)))
    at_edge = edges[peak + 1] >= -1.0 - 1e-9
    s_peak = 10 ** (0.5 * (edges[peak] + edges[peak + 1]))
    print(f"{name:18s} {A.n_obs:7d} {a.k:2d} {u.size:11d} {low.size:18d} "
          f"{'none' if at_edge else f'{100 * s_peak:.2f}':>10s} {100 * np.median(low) if low.size else float('nan'):21.2f} "
          f"{100 * s_model:13.2f} {e:6.0f}   per-cluster peaks %: {np.round(100 * np.sort(per), 2).tolist()}")
