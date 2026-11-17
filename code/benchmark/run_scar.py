#!/usr/bin/env python
"""scAR (Novartis) on one library, following its mRNA denoising tutorial.

    python run_scar.py --input out/prep/<name>.raw_v3.h5 --cells out/prep/<name>.cells.txt \
        --output out/scar/<name>.h5ad

Steps, as in the tutorial: estimate the ambient profile from cell-free droplets with
setup_anndata(prob=0.995), train model(feature_type="mRNA", sparsity=1) for 200 epochs at
batch size 64, then take native_counts. Genes with no counts in any cell are left out of the
network and written back as zeros, which changes nothing in the output.
"""

from __future__ import annotations

import argparse
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_10x_v3(path) -> ad.AnnData:
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(s) for s in g["shape"][:])
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape).T.tocsr()
        dec = lambda a: np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])  # noqa: E731
        var = pd.DataFrame({"gene_id": dec(g["features/id"][:]), "gene_name": dec(g["features/name"][:]),
                            "genome": dec(g["features/genome"][:])})
        var["feature_types"] = "Gene Expression"
        var.index = var["gene_id"].to_numpy()
        return ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=dec(g["barcodes"][:])), var=var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    import torch
    from scar import model, setup_anndata

    t0 = time.perf_counter()
    raw = read_10x_v3(a.input)
    cells = [x.strip() for x in open(a.cells) if x.strip()]
    filt = raw[cells].copy()
    keep = np.asarray(filt.X.sum(0)).ravel() > 0
    raw_k, filt_k = raw[:, keep].copy(), filt[:, keep].copy()
    print(f"[scar] {raw.n_obs} droplets, {filt.n_obs} cells, {int(keep.sum())} of {raw.n_vars} genes with counts in cells",
          flush=True)

    setup_anndata(adata=filt_k, raw_adata=raw_k, prob=0.995, kneeplot=False)
    key = [k for k in filt_k.uns if str(k).startswith("ambient_profile")]
    if not key:
        raise SystemExit("[scar] setup_anndata did not store an ambient profile")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = model(raw_count=filt_k, ambient_profile=filt_k.uns[key[0]], feature_type="mRNA", sparsity=1, device=device)
    m.train(epochs=a.epochs, batch_size=a.batch_size, verbose=True)
    m.inference()
    den = m.native_counts
    den = sp.csr_matrix(den) if not sp.issparse(den) else den.tocsr()

    # Back to the full gene list.
    idx = np.flatnonzero(keep)
    proj = sp.csr_matrix((np.ones(idx.size), (np.arange(idx.size), idx)), shape=(idx.size, raw.n_vars))
    out = ad.AnnData(X=(den @ proj).tocsr().astype(np.float32), obs=pd.DataFrame(index=filt.obs_names.to_numpy()),
                     var=raw.var[["gene_id", "gene_name", "genome"]].copy())
    out.write_h5ad(a.output, compression="gzip")
    removed = 1 - out.X.sum() / filt.X.sum()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0.0
    print(f"[scar] wrote {a.output}; removed {100 * removed:.3f}% of counts; {time.perf_counter() - t0:.1f} s; "
          f"{gpu}; peak GPU {peak} GB", flush=True)


if __name__ == "__main__":
    main()
