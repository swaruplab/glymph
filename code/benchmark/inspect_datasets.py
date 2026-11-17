#!/usr/bin/env python
"""Print the structure of downloaded files without loading the matrices: h5ad and 10x h5 layout,
table headers, MEX sizes. Used once per new dataset to write its ingestion rule.

    python inspect_datasets.py <file or folder> [...]
"""
import gzip
import os
import sys
import zipfile

import h5py
import numpy as np


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def h5ad(path):
    with h5py.File(path, "r") as f:
        X = f["X"]
        shape = tuple(X.attrs.get("shape", getattr(X, "shape", ())))
        enc = X.attrs.get("encoding-type", "dense")
        dtype = X["data"].dtype if isinstance(X, h5py.Group) else X.dtype
        print(f"  X {shape} {enc} {dtype}; layers {list(f['layers'].keys()) if 'layers' in f else []}; raw {'raw' in f}")
        if isinstance(X, h5py.Group):
            d = X["data"][:200000]
            print(f"  X.data sample: min {d.min()} max {d.max()} integer {bool(np.allclose(d, np.round(d)))}")
        for grp in ("obs", "var"):
            g = f[grp]
            idx = g.attrs.get("_index", "index")
            print(f"  {grp} index '{idx}': {dec(g[idx][:3])}")
            for k in g.keys():
                if k == idx:
                    continue
                v = g[k]
                if isinstance(v, h5py.Group) and "categories" in v:
                    cats = dec(v["categories"][:])
                    print(f"    {k}: categorical, {len(cats)} levels, e.g. {cats[:8]}")
                elif isinstance(v, h5py.Dataset):
                    print(f"    {k}: {v.dtype}, e.g. {dec(v[:3]) if v.dtype.kind in 'OS' else v[:3].tolist()}")
        print(f"  uns keys: {list(f['uns'].keys())[:20] if 'uns' in f else []}")


def tenx(path):
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            g = f["matrix"]
            print(f"  10x v3 layout, shape (features, barcodes) {tuple(g['shape'][:])}, nnz {g['data'].shape[0]}")
            ft = g["features"]
            print(f"  genomes {sorted(set(dec(ft['genome'][:])))}; types {sorted(set(dec(ft['feature_type'][:])))}; ids {dec(ft['id'][:2])}; barcodes {dec(g['barcodes'][:2])}")
        else:
            for k in f.keys():
                g = f[k]
                print(f"  10x v2 group {k}: shape {tuple(g['shape'][:])}, nnz {g['data'].shape[0]}, genes {dec(g['genes'][:2])}, barcodes {dec(g['barcodes'][:2])}")


def text(path, n=4, width=260):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        shown = 0
        for line in f:
            if line.startswith("##") and shown > 2:
                continue
            print("   |", line.rstrip("\n")[:width])
            shown += 1
            if shown >= n + 8:
                break


def one(path):
    size = os.path.getsize(path)
    print(f"\n== {path}  ({size / 1e6:.1f} MB)")
    base = path[:-3] if path.endswith(".gz") else path
    try:
        if path.endswith(".zip"):
            for i in zipfile.ZipFile(path).infolist():
                print(f"  {i.filename}  {i.file_size / 1e6:.1f} MB")
        elif base.endswith(".h5ad"):
            h5ad(path)
        elif base.endswith(".h5") and not path.endswith(".gz"):
            tenx(path)
        elif base.endswith((".csv", ".tsv", ".txt", ".mtx")):
            text(path)
        else:
            print("  (not inspected)")
    except Exception as e:  # keep going: this is a survey
        print(f"  ERROR {type(e).__name__}: {e}")


for p in sys.argv[1:]:
    if os.path.isdir(p):
        for root, _, files in sorted(os.walk(p)):
            for fn in sorted(files):
                if not fn.endswith((".fastq.gz", ".fq.gz")):
                    one(os.path.join(root, fn))
    else:
        one(p)
