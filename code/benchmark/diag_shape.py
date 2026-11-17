#!/usr/bin/env python
"""What does the background inside cells look like: the soup, or the average cell?

In a human cell every mouse count is background. Pool those counts over human cells and compare
their gene profile with (a) the empty-droplet soup and (b) the pool-average cell profile, both
restricted to mouse genes. Done per depth tercile, because a depth-proportional background
should show up most in deep cells. Smaller KL = better fit.

    python diag_shape.py --raw raw.h5 --name hgmm_10k_v3
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import ordmag_threshold  # noqa: E402
from glymph.io import read_counts  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()

raw = read_counts(a.raw)
X = sp.csr_matrix(raw.X).astype(np.float64)
spg = gene_species(raw.var)
is_h, is_m = spg == "human", spg == "mouse"
per = np.stack([np.asarray(X[:, is_h].sum(1)).ravel(), np.asarray(X[:, is_m].sum(1)).ravel()], 1)
N = per.sum(1)
T, _ = ordmag_threshold(np.sort(N)[::-1])
pur = per.max(1) / np.maximum(N, 1)
cell = (N >= T) & (pur >= 0.9)
own = per.argmax(1)
empty = (N >= 5) & (N <= T / 10)
soup = np.asarray(X[empty].sum(0)).ravel()
chibar = np.asarray(X[N >= T].sum(0)).ravel()


def kl(p, q):
    p, q = p / p.sum(), (q + 1e-12) / (q + 1e-12).sum()
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def entropy_floor(p, n):
    """Expected KL of an n-count multinomial sample from p against p itself (sampling noise)."""
    return (np.count_nonzero(p) - 1) / (2.0 * n)


print(f"{a.name}: threshold {T:.0f}; {int(cell.sum())} cells; empties' mean {N[empty].mean():.0f} UMIs")
print("cells    depth_tercile  median_N  cross_UMIs   KL_to_soup  KL_to_avg_cell  noise_floor  best_mix_soup_weight  KL_best_mix")
for sp_i, name, other in ((0, "human", is_m), (1, "mouse", is_h)):
    idx = np.flatnonzero(cell & (own == sp_i))
    q = np.quantile(N[idx], [0, 1 / 3, 2 / 3, 1])
    for t in range(3):
        sel = idx[(N[idx] >= q[t]) & (N[idx] <= q[t + 1])]
        cross = np.asarray(X[sel][:, other].sum(0)).ravel()
        s_o, c_o = soup[other], chibar[other]
        ws = np.linspace(0, 1, 21)
        kls = [kl(cross, w * s_o / s_o.sum() + (1 - w) * c_o / c_o.sum()) for w in ws]
        b = int(np.argmin(kls))
        print(f"{name:8s} {t + 1:13d} {int(np.median(N[sel])):9d} {int(cross.sum()):11d} {kl(cross, s_o):12.4f} "
              f"{kl(cross, c_o):15.4f} {entropy_floor(cross, cross.sum()):12.4f} {ws[b]:21.2f} {kls[b]:12.4f}")
