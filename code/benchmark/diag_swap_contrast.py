#!/usr/bin/env python
"""Can the swapped (cell-profile) share of the background be measured without labels? (development)

Background of gene g in cluster c = kappa * A + kappa_d * A2 + s * D, where A and A2 carry the soup profile and D the
average cell profile. Summed over a cell's silent genes the last two cannot be told apart (both grow with depth), so
the per-cell fit leaves s where it started. Gene by gene they can: a gene that cells hold more of than the soup does
(nuclear and mitochondrial RNA) has more background than the soup predicts exactly when s > 0.

Label-free iteration tested here, on the cluster sums of a debug dump (GLYMPH_DEBUG_DUMP): take the candidate pairs
(rate at most a fifth of the top cluster, enough background expected), find the lowest ridge of observed over
expected, fit the three amounts by Poisson regression on the pairs of that ridge, repeat. With --raw of a species
mixture each step is checked against the cross-species counts of the pure clusters (the truth).

    python diag_swap_contrast.py --dump out_dbg/LIB.pool0.pt [--raw data/std/LIB.h5ad] [--iters 12]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.exclusion import _leftmost_peak, _poisson_plane  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--raw")
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--window", type=float, default=0.15, help="dex above the ridge that still counts as the ridge")
a = ap.parse_args()
d = torch.load(a.dump, map_location="cpu", weights_only=False)
f64 = torch.float64
X, A, A2 = d["Xc"].to(f64), d["A"].to(f64), d["A2"].to(f64)
D = d["Nc"].to(f64).unsqueeze(1) * d["chi"].to(f64).unsqueeze(0)
big, keep, rate, rmax = d["big"], d["keep_c"], d["rate"], d["rmax"]
need, delta, tail = d["need"], d["delta"], d["tail"]
k, G = X.shape
truth = None
if a.raw:
    from species_mix_metrics import gene_species, read_counts

    raw = read_counts(a.raw)
    gi = np.load(a.dump[: a.dump.rindex(".pool")] + ".index.npz")["genes"]
    spg = gene_species(raw.var)[gi]
    hum, mus = torch.tensor(spg == "human"), torch.tensor(spg == "mouse")
    xh, xm = X[:, hum].sum(1), X[:, mus].sum(1)
    is_h = xh >= xm
    pur = torch.maximum(xh, xm) / (xh + xm).clamp_min(1)
    cross = torch.where(is_h.unsqueeze(1), mus.unsqueeze(0), hum.unsqueeze(0))
    pure = (pur >= 0.95) & big & keep
    art = torch.zeros(G, dtype=torch.bool)
    B0 = d["theta"][0] * A + d["theta"][1] * A2 + d["theta"][2] * D
    for o, og in ((True, mus), (False, hum)):
        cl = pure & (is_h == o)
        if cl.any():
            r_ = X[cl].sum(0) / (d["lv"] * B0[cl].sum(0)).clamp_min(1e-9)
            art |= og & (r_ > 4) & (X[cl].sum(0) >= 20)
    truth = cross & pure.unsqueeze(1) & ~art.unsqueeze(0)
    yt, Zt = X[truth], torch.stack([A[truth], A2[truth], D[truth]], 1)
    th_or = _poisson_plane(yt, Zt, torch.ones_like(yt), torch.tensor([1.0, 0.1, 0.005], dtype=f64), torch.zeros(3, dtype=f64), torch.tensor([50.0, 50.0, 0.2], dtype=f64), iters=3000)
    print(f"{os.path.basename(a.dump)}: fitted on the cross-species counts of the pure clusters: {[round(float(v), 4) for v in th_or]}")


def report(th, tag):
    B = th[0] * A + th[1] * A2 + th[2] * D
    out = f"{tag}: kappa {float(th[0]):.3f}, kappa_d {float(th[1]):.3f}, s {float(th[2]):.4f}"
    cand = (big & keep).unsqueeze(1) & (rate <= delta * rmax.unsqueeze(0))
    if truth is not None:
        lam = B.clamp_min(1e-12)
        tested = (B >= need) & (X <= 10 ** 0.15 * B)
        rare = (B < need) & ((X < 0.5) | (torch.special.gammainc(X.clamp_min(1e-6), lam) >= tail))
        sil = cand & (tested | rare)
        out += f"; cross-species counts of the pure clusters: expected {float((B * truth).sum()) / float((X * truth).sum()):.3f} of observed, silent at level 1 {100 * float((X * (truth & sil)).sum()) / float((X * truth).sum()):.1f}%"
        # the other side: counts of each pure cluster's own species that the test would call silent (cannot be checked, shown for size)
        own = (~cross) & pure.unsqueeze(1)
        out += f"; own-species counts called silent {100 * float((X * (own & sil)).sum()) / float((X * own).sum()):.2f}%"
    print(out)


th = d["theta"].to(f64).clone()
lo, hi = torch.zeros(3, dtype=f64), torch.tensor([50.0, 50.0, 0.2], dtype=f64)
report(th, "Glymph's fit")
if truth is not None:
    report(th_or, "truth fit   ")
for it in range(a.iters):
    B = th[0] * A + th[1] * A2 + th[2] * D
    rho = X / B.clamp_min(1e-12)
    cand = (big & keep).unsqueeze(1) & (rate <= delta * rmax.unsqueeze(0)) & (B >= need)
    level, _ = _leftmost_peak(torch.log10(rho[cand].clamp_min(1e-12)), frac=0.1)
    if level is None:
        print("no ridge"); break
    P = cand & (rho <= level * 10 ** a.window)
    y, Z = X[P], torch.stack([A[P], A2[P], D[P]], 1)
    new = _poisson_plane(y, Z, torch.ones_like(y), th.clamp_min(1e-5), lo, hi, iters=3000)
    report(new, f"  step {it + 1:2d} (ridge {level:.2f}, {int(P.sum())} of {int(cand.sum())} candidate pairs)")
    if bool(((new - th).abs() <= 0.01 * th.clamp_min(1e-4)).all()):
        th = new
        break
    th = new
