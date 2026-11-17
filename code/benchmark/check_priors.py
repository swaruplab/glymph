#!/usr/bin/env python
"""Readiness check for a new library: can it be read, and where do the thresholds land?

    python check_priors.py --raw raw.h5 [--filtered filtered.h5] [--layout auto] [--plot rank.png]

Prints the barcode count, the depth quantiles, the cell threshold, the empty-droplet level and
the number of barcodes above the threshold, beside the vendor's cell count when a filtered
matrix is given. No model is fitted.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import estimate_priors, plot_priors  # noqa: E402
from glymph.io import read_counts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--filtered")
ap.add_argument("--layout", default="auto")
ap.add_argument("--plot")
ap.add_argument("--name", default="")
a = ap.parse_args()

raw = read_counts(a.raw, a.layout)
X = sp.csr_matrix(raw.X)
N = np.asarray(X.sum(1)).ravel()
pri = estimate_priors(X)
d = pri.diagnostics
q = np.quantile(N, [0.5, 0.9, 0.99, 0.999])
line = (f"{a.name or os.path.basename(a.raw)} [{d['threshold_method']}"
        + (f", cliff {d['bead_cliff']:.0f}" if d.get("bead_cliff") else "") + f"]: {X.shape[0]} barcodes x {X.shape[1]} genes, {int(N.sum())} counts; "
        f"depth q50/q90/q99/q99.9 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}; cell threshold {d['cell_threshold']:.0f} "
        f"({d['n_cells_prior']} barcodes above); empty level {d['empty_level']:.0f}; background window top {d['empty_max']:.0f}; "
        f"{d['n_background_barcodes']} background barcodes; {d['n_analyzed']} analysed")
if a.filtered:
    import h5py

    with h5py.File(a.filtered, "r") as f:
        fb = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["matrix/barcodes"][:]])
    inraw = raw.obs_names.isin(fb)
    line += (f"; vendor cells {len(fb)} (median depth {np.median(N[inraw]):.0f}, min {N[inraw].min():.0f}); "
             f"vendor cells above our threshold {int((N[inraw] >= d['cell_threshold']).sum())}")
print(line)
if a.plot:
    plot_priors(X, pri, a.plot)
