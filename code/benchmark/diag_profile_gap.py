#!/usr/bin/env python
"""How far is the background that reaches cells from the profile of the empty droplets, gene by gene?

Read-only feasibility check for a per-gene background profile measured inside cells. In a species
mixture the other species' genes show the background of a cell directly. For the cells of each
species, the counts on the other species' genes are compared with
  soup     the other-species part of the low-count barcodes (what Glymph assumes for ambient RNA)
  mix      the best blend of soup and the other species' average cell profile (ambient plus swapped)
  factor   mix times a per-gene factor learned on one random half of the cells (shrunk toward 1),
           scored on the other half
by the share of background counts each profile puts on the wrong genes (total variation distance).
"floor" is that share for a perfect profile at the same number of counts (sampling noise alone).
Also: the split-half correlation of the per-gene log ratio observed / mix, which says how much of the
gene-to-gene deviation is a stable property of the gene.

    python diag_profile_gap.py --name hgmm_5k_v3 --raw data/std/hgmm_5k_v3.h5ad --cells out/prep/hgmm_5k_v3.cells.txt
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glymph.io import read_counts  # noqa: E402
from species_mix_metrics import gene_species  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--alpha", type=float, default=10.0, help="prior counts of the per-gene factor")
a = ap.parse_args()
rng = np.random.default_rng(0)
raw = read_counts(a.raw)
X = sp.csr_matrix(raw.X)
spg = gene_species(raw.var)
hm = np.stack([spg == "human", spg == "mouse"]).astype(np.float64)
per_sp = np.asarray(X @ hm.T)
N = per_sp.sum(1)
listed = raw.obs_names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(N[listed], 0.01))
purity = per_sp.max(1) / np.maximum(N, 1)
ev = np.flatnonzero(listed & (purity >= 0.9))
own = per_sp.argmax(1)[ev]
bg = (N >= 5) & (N <= T / 10.0)
soup_g = np.asarray(X[bg].sum(0)).ravel().astype(np.float64)


def tv(counts, prof):
    return 0.5 * float(np.abs(counts / counts.sum() - prof / prof.sum()).sum())


def floor(M, prof):
    p = prof / prof.sum()
    return float(np.mean([tv(rng.multinomial(int(M), p).astype(float), p) for _ in range(3)]))


for o, who in ((0, "human"), (1, "mouse")):
    other = hm[1 - o] > 0
    cells_o, cells_x = ev[own == o], ev[own == 1 - o]
    if cells_o.size < 100 or cells_x.size < 100:
        continue
    cross = np.asarray(X[cells_o].sum(0)).ravel() * other
    src = np.asarray(X[cells_x].sum(0)).ravel() * other
    soup = soup_g * other
    # alignment artifacts out, by the scorer's rule
    cs = cross / max(cross.sum(), 1.0)
    ref = np.maximum(src / max(src.sum(), 1.0), soup / max(soup.sum(), 1.0))
    keep = other & ~((cs >= 0.005) & (cs > 10 * ref))
    cross, src, soup = cross * keep, src * keep, soup * keep
    ps, pc = soup / soup.sum(), src / src.sum()
    M = cross.sum()
    ws = np.linspace(0, 1, 21)
    tvs = [tv(cross, w * ps + (1 - w) * pc) for w in ws]
    w = float(ws[int(np.argmin(tvs))])
    pm = w * ps + (1 - w) * pc
    half = rng.random(cells_o.size) < 0.5
    cA = np.asarray(X[cells_o[half]].sum(0)).ravel() * keep
    cB = np.asarray(X[cells_o[~half]].sum(0)).ravel() * keep
    phi = (a.alpha + cA) / (a.alpha + cA.sum() * pm)
    big = (cA.sum() * pm >= 20) & (cB.sum() * pm >= 20)
    rA = np.log2((cA[big] + 0.5) / (cA.sum() * pm[big] + 0.5))
    rB = np.log2((cB[big] + 0.5) / (cB.sum() * pm[big] + 0.5))
    rel = float(np.corrcoef(rA, rB)[0, 1]) if big.sum() > 20 else float("nan")
    sd = float(np.sqrt(max(np.cov(rA, rB)[0, 1], 0.0))) if big.sum() > 20 else float("nan")
    print(f"{a.name}\t{who} cells {cells_o.size}\tbackground counts on the other species' genes {int(M)}\t"
          f"wrong-gene share: soup {100 * tv(cB, ps):.1f}%, mix {100 * tv(cB, pm):.1f}% (soup weight {w:.2f}), "
          f"per-gene factor from the other half {100 * tv(cB, pm * phi):.1f}%, floor {100 * floor(cB.sum(), pm * phi):.1f}%\t"
          f"genes compared {int(big.sum())}, split-half r {rel:.2f}, stable spread {sd:.2f} log2 (x{2 ** sd:.2f})")
