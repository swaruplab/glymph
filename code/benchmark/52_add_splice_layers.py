#!/usr/bin/env python
"""Add the spliced / unspliced molecule counts to a standard input file as layers.

    python 52_add_splice_layers.py --raw data/std/kidney_nuc2.h5ad --splice data/splice/kidney_nuc2 --out data/std_splice/kidney_nuc2.h5ad

layers["unspliced"]: molecules with intronic sequence. layers["spliced"]: all others (exonic only, with or without a
junction). X is left as it is; the layers come from the alignment file and need not add up to X exactly.
"""
import argparse
import os

import anndata as ad
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--splice", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
x = ad.read_h5ad(a.raw)
U, S, A = (sp.load_npz(f"{a.splice}.{k}.npz").tocsr().astype(np.int32) for k in ("unspliced", "spliced", "ambiguous"))
assert U.shape == x.shape, (U.shape, x.shape)
x.layers["unspliced"] = U
x.layers["spliced"] = (S + A).tocsr()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
x.write_h5ad(a.out, compression="gzip")
n, u = x.X.sum(), U.sum()
print(f"{a.out}: {x.shape[0]} barcodes, {x.shape[1]} genes; molecules in the layers over counts in X {(u + S.sum() + A.sum()) / n:.3f}; unspliced share {u / (u + S.sum() + A.sum()):.3f}")
