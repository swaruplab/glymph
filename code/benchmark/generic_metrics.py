#!/usr/bin/env python
"""Scores that need no species or genotype truth, for every library and tool.

  Count integrity   entries above the raw count, negative or fractional entries, counts added.
  Amount            share of counts removed from the evaluation cells; cells the tool did not return.
  Marker leakage    For groups of cells, exclusive markers are genes whose share of counts in one
                    group is at least 20 times their share in all other cells (raw data, at most
                    50 per group). Counts of a group's markers in other groups' cells are mostly
                    background, so
                      marker_removed_off   share of those counts removed (higher is better)
                      marker_kept_on       share of marker counts kept in the group's own cells
                    Groups are the labels attached at ingestion when they are a design fact (the
                    tissue loaded in each split-pool well); otherwise k-means clusters of the raw
                    cells, the same for every tool. With clusters the off-target counts include
                    some real low expression, so read the two numbers as a pair, tool against tool.

    python generic_metrics.py --name bd_spleen --raw std/bd_spleen.h5ad --cells out/prep/bd_spleen.cells.txt \
        --tool glymph=out/glymph/bd_spleen.h5ad --tool cellbender=out/cellbender/bd_spleen/bd_spleen_filtered.h5 --out out/generic/bd_spleen.tsv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.io import read_counts  # noqa: E402


def groups_from_clusters(X, k=12, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    tot = np.asarray(X.sum(1)).ravel()
    Z = sp.diags(1e4 / np.maximum(tot, 1)) @ X
    Z.data = np.log1p(Z.data)
    top = np.argsort(-np.asarray(X.sum(0)).ravel())[:4000]
    emb = TruncatedSVD(n_components=min(30, len(top) - 1), random_state=seed).fit_transform(Z[:, top])
    return KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(emb).astype(str)


def exclusive_markers(X, grp, fold=20.0, min_counts=200, top=50, min_cells=100):
    out = {}
    names, counts = np.unique(grp, return_counts=True)
    total = np.asarray(X.sum(0)).ravel()
    N = float(total.sum())
    for g, n in zip(names, counts):
        if g == "" or n < min_cells or n > len(grp) - min_cells:
            continue
        sel = grp == g
        own = np.asarray(X[sel].sum(0)).ravel()
        n_own = float(own.sum())
        rest, n_rest = total - own, N - n_own
        if n_own <= 0 or n_rest <= 0:
            continue
        ok = (own >= min_counts) & (own / n_own >= fold * (rest + 1.0) / n_rest)
        idx = np.flatnonzero(ok)
        if idx.size:
            out[g] = idx[np.argsort(-own[idx])[:top]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tool", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-min-groups", type=int, default=2)
    a = ap.parse_args()

    raw = read_counts(a.raw)
    listed = [x.strip() for x in open(a.cells) if x.strip()]
    idx = raw.obs_names.get_indexer(listed)
    idx = idx[idx >= 0]
    names = raw.obs_names[idx].to_numpy()
    Xe = sp.csr_matrix(raw.X[idx]).astype(np.float64)
    how = "none"
    grp = None
    if "label" in raw.obs:
        lab = raw.obs["label"].astype(str).to_numpy()[idx]
        vc = pd.Series(lab[lab != ""]).value_counts()
        if (vc >= 100).sum() >= a.label_min_groups:
            grp, how = lab, "labels"
    markers = exclusive_markers(Xe, grp) if grp is not None else {}
    if len(markers) < 2 and len(idx) >= 1500:
        grp, how = groups_from_clusters(Xe), "clusters"
        markers = exclusive_markers(Xe, grp)
    n_mk = int(sum(len(v) for v in markers.values()))
    print(f"[generic] {a.name}: {len(idx)} evaluation cells, groups from {how}, {len(markers)} groups with markers, {n_mk} markers")

    rows = []
    for tool, path in [("raw", None)] + [t.split("=", 1) for t in a.tool]:
        if path is None:
            present, Y = np.ones(len(idx), bool), Xe
        else:
            if not os.path.exists(path):
                continue
            t = read_counts(path, integer=False)
            gi = pd.Index(raw.var["gene_id"].astype(str)).get_indexer(t.var["gene_id"].astype(str))
            ok = gi >= 0
            proj = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), gi[ok])), shape=(t.n_vars, raw.n_vars))
            A = sp.csr_matrix(t.X).astype(np.float64) @ proj
            ri = t.obs_names.get_indexer(names)
            present = ri >= 0
            Y = sp.csr_matrix(A[ri[present]])
        Xp = Xe[present]
        nb, na_ = float(Xp.sum()), float(Y.sum())
        diff = (Y - Xp).tocsr()
        row = {"dataset": a.name, "tool": tool, "n_eval_cells": int(len(idx)), "n_missing": int(len(idx) - present.sum()),
               "removed_pct": 100 * (nb - na_) / max(nb, 1), "groups_from": how, "n_marker_groups": len(markers), "n_markers": n_mk,
               "n_above_raw": int((diff.data > 1e-6).sum()), "added_counts": float(diff.data[diff.data > 0].sum()),
               "n_negative": int((Y.data < 0).sum()), "n_noninteger": int((np.abs(Y.data - np.round(Y.data)) > 1e-6).sum())}
        if markers:
            g_p = grp[present]
            off_b = off_a = on_b = on_a = 0.0
            for g, cols in markers.items():
                own = g_p == g
                mb = np.asarray(Xp[:, cols].sum(1)).ravel()
                ma = np.asarray(Y[:, cols].sum(1)).ravel()
                off_b += mb[~own].sum(); off_a += ma[~own].sum(); on_b += mb[own].sum(); on_a += ma[own].sum()
            row.update({"marker_off_pct_before": 100 * off_b / max(nb, 1), "marker_removed_off": 1 - off_a / max(off_b, 1.0),
                        "marker_kept_on": on_a / max(on_b, 1.0)})
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False, float_format="%.5g")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
