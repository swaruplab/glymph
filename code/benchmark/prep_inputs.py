#!/usr/bin/env python
"""Write the shared inputs the baseline tools read, one pair of files per library.

    <outdir>/<name>.raw_v3.h5   the raw matrix in the Cell Ranger v3 HDF5 layout, gene expression
                                only, barcodes with zero counts dropped. Cell Ranger v2 files (one
                                group per genome) become one matrix with a genome tag per gene.
    <outdir>/<name>.cells.txt   barcodes handed to tools that need cell calls (SoupX, DecontX,
                                scAR). Tier 1: total counts at or above the order-of-magnitude
                                threshold, the rule the metrics script uses to pick evaluation
                                cells. Tier 2 (--filtered): Cell Ranger's filtered barcodes.

    python prep_inputs.py --name hgmm_6k_v2 --raw raw_gene_bc_matrices_h5.h5 --outdir out/prep
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import ordmag_threshold  # noqa: E402
from glymph.io import read_counts  # noqa: E402


def write_10x_v3(path, X, barcodes, gene_id, gene_name, genome):
    """X: cells x genes CSR of integer counts."""
    M = sp.csc_matrix(X.T)  # genes x cells, compressed over cells, as Cell Ranger stores it
    M.sort_indices()
    tmp = str(path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("matrix")
        g["barcodes"] = np.asarray(barcodes, dtype="S")
        g["data"] = M.data.astype(np.int32)
        g["indices"] = M.indices.astype(np.int64)
        g["indptr"] = M.indptr.astype(np.int64)
        g["shape"] = np.array(M.shape, dtype=np.int32)
        t = g.create_group("features")
        t["id"] = np.asarray(gene_id, dtype="S")
        t["name"] = np.asarray(gene_name, dtype="S")
        t["genome"] = np.asarray(genome, dtype="S")
        t["feature_type"] = np.array([b"Gene Expression"] * len(gene_id))
        t["_all_tag_keys"] = np.array([b"genome"])
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--filtered", help="Cell Ranger filtered matrix; its barcodes become the cell list instead of the count threshold")
    ap.add_argument("--cells", help="file with the vendor pipeline's cell barcodes, one per line (BD, STARsolo); used as the cell list")
    ap.add_argument("--knee", action="store_true",
                    help="without vendor calls, take barcodes above the gap between cells and background "
                         "(glymph.background confident threshold) instead of the order-of-magnitude threshold, which fails on split-pool data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    raw = read_counts(a.raw)
    X = sp.csr_matrix(raw.X)
    N = np.asarray(X.sum(1)).ravel()
    T, _ = ordmag_threshold(np.sort(N)[::-1])
    if a.filtered:
        with h5py.File(a.filtered, "r") as f:
            fb = [x.decode() if isinstance(x, bytes) else str(x) for x in f["matrix/barcodes"][:]]
        cells = raw.obs_names[raw.obs_names.isin(fb)]
        how = f"{len(cells)} of {len(fb)} Cell Ranger filtered barcodes"
    elif a.cells:
        fb = [x.strip() for x in open(a.cells) if x.strip()]
        cells = raw.obs_names[raw.obs_names.isin(fb)]
        how = f"{len(cells)} of {len(fb)} vendor-called barcodes"
    elif a.knee:
        from glymph.background import estimate_priors

        d = estimate_priors(X).diagnostics
        T = d["confident_threshold"]
        cells = raw.obs_names[N >= T]
        how = f"{len(cells)} barcodes above the cell/background gap ({d['threshold_method']})"
    else:
        cells = raw.obs_names[N >= T]
        how = f"{len(cells)} barcodes at or above the threshold"
    v = raw.var
    write_10x_v3(os.path.join(a.outdir, f"{a.name}.raw_v3.h5"), X, raw.obs_names.to_numpy(),
                 v["gene_id"].astype(str).to_numpy(), v["gene_name"].astype(str).to_numpy(),
                 v["genome"].astype(str).to_numpy())
    with open(os.path.join(a.outdir, f"{a.name}.cells.txt"), "w") as f:
        f.write("\n".join(cells) + "\n")
    print(f"[prep] {a.name}: {X.shape[0]} barcodes x {X.shape[1]} genes, threshold {T:.0f}, cells: {how}")


if __name__ == "__main__":
    main()
