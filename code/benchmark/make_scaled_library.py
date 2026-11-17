#!/usr/bin/env python
"""Build a larger library from a real one, for run-time and memory scaling tests only.

    python make_scaled_library.py --raw raw.h5 --copies 8 --out scaled_x8.h5

Each copy is the real raw matrix thinned binomially to 90% of its counts with its own seed, so
copies are not identical, and its barcodes get a suffix. The result has the real library's rank
curve and gene profiles and `copies` times as many barcodes. It is not a biological dataset:
use it to measure time and memory, never accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glymph.io import read_counts  # noqa: E402
from prep_inputs import write_10x_v3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--copies", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keep", type=float, default=0.9)
a = ap.parse_args()

raw = read_counts(a.raw)
X = sp.csr_matrix(raw.X)
mats, bcs = [], []
for k in range(a.copies):
    rng = np.random.default_rng(k)
    M = X.copy()
    M.data = rng.binomial(M.data, a.keep).astype(M.data.dtype)
    M.eliminate_zeros()
    keep = np.diff(M.indptr) > 0
    mats.append(M[keep])
    bcs.append(np.char.add(raw.obs_names.to_numpy().astype(str)[keep], f"_c{k}"))
big = sp.vstack(mats, format="csr")
v = raw.var
write_10x_v3(a.out, big, np.concatenate(bcs), v["gene_id"].astype(str).to_numpy(), v["gene_name"].astype(str).to_numpy(),
             v["genome"].astype(str).to_numpy())
print(f"[scale] {a.copies} copies: {big.shape[0]} barcodes x {big.shape[1]} genes, {int(big.sum())} counts -> {a.out}")
