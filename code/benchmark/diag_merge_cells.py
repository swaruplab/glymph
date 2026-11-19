#!/usr/bin/env python
"""Cell by cell: is a cluster that looks like a larger cluster plus background-like RNA a set of dirty cells or a cell type?

For pairs (cluster, larger cluster) named in the report of a run with the silent-test dump: each cell's share of
counts on the larger cluster's silent genes, in the cluster and in the larger cluster. Dirty cells are the upper
tail of one distribution (no gap between the two, wide spread inside the cluster); a cell type is a population of
its own (a gap, a stable share).

    python diag_merge_cells.py --raw data/std/LIB.h5ad --dump out_dbg/TAG/LIB.pool0.pt --report out_dbg/TAG/LIB.report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species, read_counts  # noqa: E402

ap = argparse.ArgumentParser()
for k_ in ("raw", "dump", "report"):
    ap.add_argument("--" + k_, required=True)
a = ap.parse_args()


def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                out.append(v)
            find(v, key, out)
    elif isinstance(o, list):
        for v in o:
            find(v, key, out)
    return out


rep = json.load(open(a.report))
pairs = [(r, "given the larger cluster's set") for r in sum(find(rep, "clusters_given_a_larger_clusters_silent_genes", []), [])]
pairs += [(r, "kept apart") for r in sum(find(rep, "clusters_far_above_background_kept_apart", []), [])[:6]]
raw = read_counts(a.raw)
idx = np.load(a.dump[: a.dump.rindex(".pool")] + ".index.npz")
gi, analyzed = idx["genes"], idx["analyzed"]
d = torch.load(a.dump, map_location="cpu", weights_only=False)
rows, lab, wide = d["rows"].numpy(), d["lab"].numpy(), d["wide"].numpy()
merged = d.get("merged_into", {})
X = sp.csr_matrix(raw.X)[analyzed[rows]][:, gi].astype(np.float64)
N = np.asarray(X.sum(1)).ravel()
try:
    spg = gene_species(raw.var)[gi]
    hum, mus = spg == "human", spg == "mouse"
    xh, xm = np.asarray(X[:, hum].sum(1)).ravel(), np.asarray(X[:, mus].sum(1)).ravel()
    minor = np.minimum(xh, xm) / np.maximum(xh + xm, 1) if hum.any() and mus.any() else None
except Exception:  # noqa: BLE001
    minor = None
print(f"{os.path.basename(a.dump)}: {len(rows)} calibration cells, {wide.shape[0]} clusters")
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
for r, what in pairs:
    c, a_ = r["cluster"], r["like"]
    root = a_
    while root in merged:
        root = merged[root]
    S = wide[root]
    y = np.asarray(X[:, S].sum(1)).ravel() / np.maximum(N, 1)
    yc, ya = y[lab == c], y[lab == a_]
    if yc.size < 5 or ya.size < 5:
        continue
    lc = np.log(np.maximum(yc, 1e-5))
    line = (f"cluster {c} ({yc.size} cells) against {a_} ({ya.size} cells), {what}; share {r['share_of_counts']}, {r['over_background']} times background, match {r['match']}\n"
            f"    share of each cell's counts on the larger cluster's silent genes, 1/5/25/50/75/95/99%: cluster " + " ".join(f"{v:.4f}" for v in np.quantile(yc, q))
            + " | larger cluster " + " ".join(f"{v:.4f}" for v in np.quantile(ya, q))
            + f"\n    gap (5% of the cluster over 99% of the larger cluster) {np.quantile(yc, 0.05) / max(np.quantile(ya, 0.99), 1e-6):.2f}; spread inside the cluster (sd of log share) {lc.std():.2f}; "
              f"95% over 5% {np.quantile(yc, 0.95) / max(np.quantile(yc, 0.05), 1e-6):.1f}")
    if minor is not None:
        line += f"; minor-species share of the cluster's cells, median {np.median(minor[lab == c]):.3f} (larger cluster {np.median(minor[lab == a_]):.4f})"
    print(line)
