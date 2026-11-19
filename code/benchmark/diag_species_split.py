#!/usr/bin/env python
"""How is the background inside cells split between the two species? (species mixtures; raw data only)

The benchmark's truth for same-species background takes each cell's cross-species counts times the species split of
the empty partitions. That holds if the background in a cell has the empty partitions' split. It can be checked where
the background is visible: human RNA in mouse cells against mouse RNA in human cells. For each species of cell the
cross-species counts are regressed on depth (constant plus slope, Poisson), which gives the background a cell of a
given depth receives from the other species; the two together give the split of the background inside cells, to
compare with the split of the empty partitions and with the split of the cells' own RNA (what swapped molecules
would follow).

    python diag_species_split.py --name LIB --raw data/std/LIB.h5ad --cells out/prep/LIB.cells.txt
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species, read_counts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--purity", type=float, default=0.9)
a = ap.parse_args()
raw = read_counts(a.raw)
X = sp.csr_matrix(raw.X)
spg = gene_species(raw.var)
hm = np.stack([spg == "human", spg == "mouse"]).astype(np.float64)
per = np.asarray(X @ hm.T)
N = per.sum(1)
listed = raw.obs_names.isin([x.strip() for x in open(a.cells) if x.strip()])
T = float(np.quantile(N[listed], 0.01))
bg = (N >= 5) & (N <= T / 10.0)
soup = per[bg].sum(0) / per[bg].sum()
pur = per.max(1) / np.maximum(N, 1)
ev = listed & (pur >= a.purity)
own = per.argmax(1)


def fit(n, y):  # y ~ Poisson(a + b n), a, b >= 0, by EM
    th = np.array([max(y.mean() * 0.5, 1e-6), max(y.mean() * 0.5 / n.mean(), 1e-12)])
    Z = np.stack([np.ones_like(n), n], 1)
    col = Z.sum(0)
    for _ in range(5000):
        mu = np.maximum(Z @ th, 1e-12)
        new = th * ((y / mu)[:, None] * Z).sum(0) / col
        if np.all(np.abs(new - th) <= 1e-7 * np.maximum(th, 1e-12)):
            th = new
            break
        th = new
    return th


res = {}
for o, nm in ((0, "human"), (1, "mouse")):
    c = ev & (own == o)
    n, y = N[c], per[c, 1 - o]
    # the cleanest nine tenths of the cells, so that a few doublets below the purity cut do not set the slope
    keep = y / np.maximum(n, 1) <= np.quantile(y / np.maximum(n, 1), 0.9)
    res[o] = (fit(n[keep], y[keep]), int(c.sum()), float(np.median(n)), float(y.sum() / n.sum()))
cells_split = np.array([per[ev & (own == 0), 0].sum(), per[ev & (own == 1), 1].sum()])
cells_split = cells_split / cells_split.sum()
print(f"{a.name}: empty partitions {100 * soup[0]:.1f}% human; own RNA of the called cells {100 * cells_split[0]:.1f}% human; "
      f"{res[0][1]} human and {res[1][1]} mouse cells (median depth {res[0][2]:.0f} and {res[1][2]:.0f})")
(am, bm), (ah, bh) = res[0][0], res[1][0]  # mouse background in human cells; human background in mouse cells
print(f"  mouse RNA in human cells: {am:.2f} counts + {100 * bm:.3f}% of depth ({100 * res[0][3]:.2f}% of their counts)")
print(f"  human RNA in mouse cells: {ah:.2f} counts + {100 * bh:.3f}% of depth ({100 * res[1][3]:.2f}% of their counts)")
print(f"  split of the constant part: {100 * ah / max(ah + am, 1e-12):.1f}% human; of the depth-proportional part: {100 * bh / max(bh + bm, 1e-12):.1f}% human")
for o, nm in ((0, "human"), (1, "mouse")):
    n = res[o][2]
    h, m = ah + bh * n, am + bm * n
    cross = m if o == 0 else h
    own_meas = h if o == 0 else m
    own_soup = cross * soup[o] / soup[1 - o]
    print(f"  a {nm} cell of median depth: background from the other species {cross:.1f} counts; from its own species, by what cells of the other species receive "
          f"{own_meas:.1f}, by the empty partitions' split {own_soup:.1f} (ratio {own_meas / max(own_soup, 1e-9):.2f})")
