#!/usr/bin/env python
"""Why is a cross-species gene not silent in a cluster? (species mixtures; development only)

Reads the dump that Glymph writes when GLYMPH_DEBUG_DUMP is set (exclusion.silent_budget) and the raw
library for gene names and species. Every cluster is given the species that holds most of its counts;
its counts on the other species' genes are background by construction, and they are split by what
the silent test said about the gene there:

  silent            in the wider silent set (stripped by the exclusion constraint)
  not a candidate   the cluster's rate is above a fifth of the top cluster's
  above the level   enough background expected to test, counts above 1.41 x level x fitted background
  rare, above       too little background expected, counts above what a Poisson at the level allows
  overruled         silent for the cluster, kept for a sub-cluster whose cells carry it (the tree)
  cluster unused    the cluster is small or was dropped from the fit

    python diag_silent_pairs.py --raw data/std/LIB.h5ad --dump out/dbg/LIB.pool0.pt [--top 15]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species, read_counts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
raw = read_counts(a.raw)
spg = gene_species(raw.var)
gname = raw.var["gene_name"].astype(str).to_numpy() if "gene_name" in raw.var else raw.var_names.astype(str).to_numpy()
d = torch.load(a.dump, map_location="cpu", weights_only=False)
X, B = d["Xc"].numpy(), d["Bc"].numpy()
k, G = X.shape
gi = np.load(a.dump[: a.dump.rindex(".pool")] + ".index.npz")["genes"]  # the genes Glymph kept, in its order
assert G == gi.size, (G, gi.size)
spg, gname = spg[gi], gname[gi]
lv, need, delta, tail = d["lv"], d["need"], d["delta"], d["tail"]
wide, big, keep = d["wide"].numpy(), d["big"].numpy(), d["keep_c"].numpy()
rate, rmax = d["rate"].numpy(), d["rmax"].numpy()
hum, mus = spg == "human", spg == "mouse"
sp_c = np.where(X[:, hum].sum(1) >= X[:, mus].sum(1), "human", "mouse")
pur = np.maximum(X[:, hum].sum(1), X[:, mus].sum(1)) / np.maximum(X[:, hum | mus].sum(1), 1)
sizes = np.bincount(d["lab"].numpy(), minlength=k)
print(f"{os.path.basename(a.dump)}: {k} clusters, level {d['level']:.2f} (test level {lv:.2f}), need {need:.0f}, theta {[round(float(t), 4) for t in d['theta']]}, cells left out of the sums {int(d['odd'].sum())} of {len(d['lab'])}")
print("cluster\tcells\tspecies\tpurity\tsilent genes\tcross-species counts\tof them silent")
cross = np.where((sp_c == "human")[:, None], mus[None, :], hum[None, :])
for c in range(k):
    xc = X[c] * cross[c]
    print(f"{c}\t{sizes[c]}\t{sp_c[c]}\t{pur[c]:.3f}\t{int(wide[c].sum())}\t{int(xc.sum())}\t{100 * (xc * wide[c]).sum() / max(xc.sum(), 1):.1f}%{'' if big[c] and keep[c] else '  (unused)'}")
cand = rate <= delta * rmax[None, :]
ratio = X / np.maximum(lv * B, 1e-12)
used = (big & keep)[:, None] & np.ones((1, G), bool)
cat = np.full((k, G), "", dtype=object)
cat[~used] = "cluster unused"
cat[used & ~cand] = "not a candidate"
cat[used & cand & (B >= need) & ~wide] = "above the level"
cat[used & cand & (B < need) & ~wide] = "rare, above"
cat[used & wide] = "silent"
Xx = X * cross
tot = Xx.sum()
print(f"\ncross-species counts in the cluster sums: {int(tot)}")
for nm in ("silent", "not a candidate", "above the level", "rare, above", "cluster unused"):
    m = cat == nm
    line = f"  {nm}: {100 * Xx[m].sum() / tot:.2f}% in {int((m & (Xx > 0)).sum())} pairs"
    if nm in ("above the level", "rare, above", "not a candidate"):
        r = ratio[m & (Xx > 0)]
        w = Xx[m & (Xx > 0)]
        edges = [0, 1.41, 2, 3, 5, 10, np.inf]
        line += "; counts by observed over level x fitted background: " + ", ".join(f"{lo:g}-{hi:g}: {100 * w[(r >= lo) & (r < hi)].sum() / tot:.2f}%" for lo, hi in zip(edges[:-1], edges[1:]))
    print(line)
if "over" in d:
    parent, over, Xf = d["parent"].numpy(), d["over"].numpy(), d["Xf"].numpy()
    crf = cross[parent]
    ov = wide[parent] & over & crf
    print(f"  overruled by a sub-cluster (the tree): {100 * (Xf * ov).sum() / tot:.2f}% in {int(ov.sum())} pairs of {len(parent)} sub-clusters")
# does the excess of a gene repeat across the clusters of the same species? (profile, not expression)
print("\ngenes above the level: is the excess the same in every cluster of the species?")
for o in ("human", "mouse"):
    cl = np.flatnonzero((sp_c == o) & big & keep)
    if cl.size < 2:
        continue
    og = mus if o == "human" else hum
    Xo, Bo = X[cl][:, og], lv * B[cl][:, og]
    tot_x, tot_b = Xo.sum(0), Bo.sum(0)
    R = tot_x / np.maximum(tot_b, 1e-9)
    ok = tot_b >= 20
    per = Xo / np.maximum(Bo, 1e-9)
    # spread across clusters: weighted sd of log ratio, clusters with at least 5 expected
    w = Bo * (Bo >= 5)
    lr = np.log(np.maximum(per, 1e-3))
    m_ = (w * lr).sum(0) / np.maximum(w.sum(0), 1e-9)
    sd = np.sqrt((w * (lr - m_) ** 2).sum(0) / np.maximum(w.sum(0), 1e-9))
    nfit = (Bo >= 5).sum(0)
    print(f"  {o} cells, {int(ok.sum())} genes of the other species with 20 or more expected counts (over {cl.size} clusters):")
    for lo, hi in ((0, 0.7), (0.7, 1.41), (1.41, 2), (2, 3), (3, 5), (5, np.inf)):
        g = ok & (R >= lo) & (R < hi)
        g2 = g & (nfit >= 2)
        print(f"    observed over level x fitted {lo:g}-{hi:g}: {int(g.sum())} genes, {100 * tot_x[g].sum() / max(tot_x.sum(), 1):.1f}% of the counts; spread across clusters (sd of log ratio), median {np.median(sd[g2]) if g2.any() else float('nan'):.2f}, 90th {np.quantile(sd[g2], 0.9) if g2.any() else float('nan'):.2f}")
    names = gname[og]
    top = np.argsort(-(tot_x - 1.41 * tot_b) * ok)[: a.top]
    print("    genes with the most counts above the level (counts, level x fitted, ratio; ratio by cluster):")
    for g in top:
        if tot_x[g] <= 1.41 * tot_b[g]:
            break
        print(f"      {names[g]}\t{int(tot_x[g])}\t{tot_b[g]:.0f}\t{R[g]:.2f}\t" + " ".join(f"{v:.1f}" for v in per[:, g]))

# What the right amounts would give: the cross-species counts of the pure clusters are background by construction,
# so a fit of the three amounts on them alone is the answer the label-free fit should reach.
if "A" in d:
    from scipy.optimize import nnls
    from scipy.stats import poisson

    A, A2, D = d["A"].numpy(), d["A2"].numpy(), d["Nc"].numpy()[:, None] * d["chi"].numpy()[None, :]
    pure = (pur >= 0.95) & big & keep
    art = np.zeros(G, bool)
    for o in ("human", "mouse"):  # genes far above the background in every pure cluster of a species: mis-aligned reads, left out
        cl = np.flatnonzero(pure & (sp_c == o))
        og = mus if o == "human" else hum
        if cl.size:
            r_ = X[cl].sum(0) / np.maximum(lv * B[cl].sum(0), 1e-9)
            art |= og & (r_ > 4) & (X[cl].sum(0) >= 20)
    m = cross & pure[:, None] & ~art[None, :]
    th0 = d["theta"].numpy()
    print(f"\npure clusters: {int(pure.sum())}; cross-species counts there {int((X * m).sum())} (genes above 4x in every pure cluster left out: {int(art.sum())}, {int((X * (cross & pure[:, None] & art[None, :])).sum())} counts)")
    y = X[m]
    Z = np.stack([A[m], A2[m], D[m]], 1)
    # Poisson fit by iteratively reweighted non-negative least squares
    th = th0.copy()
    for _ in range(30):
        mu = np.maximum(Z @ th, 1e-6)
        w = 1.0 / np.sqrt(mu)
        th, _ = nnls(Z * w[:, None], y * w)
    print(f"  amounts fitted by Glymph: {np.round(th0, 4)}; fitted on the cross-species counts themselves: {np.round(th, 4)}")
    for nm, t in (("Glymph's amounts", th0), ("amounts fitted on the cross-species counts", th)):
        Bo = A * t[0] + A2 * t[1] + D * t[2]
        for lvl_name, lvl in (("at the test level", lv), ("at level 1", 1.0)):
            lam = np.maximum(lvl * Bo, 1e-12)
            tested = (Bo >= need) & (X <= lvl * 10 ** 0.15 * Bo)
            rare = (Bo < need) & ((X < 0.5) | (poisson.sf(X - 1, lam) >= tail))
            sil = cand & (tested | rare)
            print(f"  {nm}, {lvl_name} ({lvl:.2f}): expected {int((Bo * m).sum())} against {int((X * m).sum())} observed; silent by the test: {100 * (X * (m & sil)).sum() / (X * m).sum():.1f}% of the counts")
        r = (X / np.maximum(Bo, 1e-9))[m & (Bo >= 20)]
        wts = Bo[m & (Bo >= 20)]
        lr = np.log(np.maximum(r, 1e-3))
        mu_ = np.average(lr, weights=wts)
        print(f"    pairs with 20 or more expected counts: observed over expected, weighted mean of log {mu_:.3f}, sd {np.sqrt(np.average((lr - mu_) ** 2, weights=wts)):.3f}; quantiles 10/50/90%: " + " ".join(f"{v:.2f}" for v in np.quantile(r, [0.1, 0.5, 0.9])))
    # is the misfit explained by the cell profile over the soup profile?
    Bo = A * th0[0] + A2 * th0[1] + D * th0[2]
    sel_ = m & (Bo >= 20)
    lr = np.log(np.maximum(X[sel_] / Bo[sel_], 1e-3))
    lq = np.log(np.maximum(D[sel_], 1e-12) / np.maximum(A2[sel_], 1e-12))
    print(f"  correlation of log(observed over Glymph's expected) with log(cell profile over soup profile): {np.corrcoef(lr, lq)[0, 1]:.2f} over {int(sel_.sum())} pairs")
