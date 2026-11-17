#!/usr/bin/env python
"""Tier 2 metrics: mouse kidney pooled from two subspecies (Janssen et al. 2023, GSE218853).

Truth comes from the authors, not from this project: for every cell they estimated the
background fraction from reads that carry the other subspecies' alleles (`contPerCell_binom`
in their cell_metadata), and they annotated cell types. Two kinds of score follow.

  Per-cell estimation, the endpoints of Janssen et al.
    contam_rmsle, contam_kendall   estimated fraction removed vs the genotype-based fraction,
                                   over every cell with a truth value; *_cast repeats it on
                                   M. m. castaneus cells, where the truth is measured directly
    removed_over_truth             counts removed / counts the truth says are background (ideal 1)

  Marker leakage, with their ten proximal tubule (PT) markers
    marker_removed_nonpt           share of PT-marker counts removed from non-PT cells, where
                                   those counts are overwhelmingly background (PT cells are 66
                                   to 89% of each library and dominate the soup)
    marker_kept_pt                 share of PT-marker counts kept in PT cells, where they are real
    slc34a1_nonpt_before / _after  percent of non-PT cells in which Slc34a1 is detected
                                   (at least half a count, so fractional outputs are not flattered)
    marker_log2fc                  log2 ratio of the PT-marker share of counts, PT over non-PT

Count integrity is reported as in tier 1. Cells a tool did not return are counted as missing
and left out of that tool's scores.

    python kidney_metrics.py --name kidney_nuc2 --replicate nuc2 --raw raw.h5 \
        --meta data/kidney_strainmix/cell_metadata.tsv.gz --markers data/kidney_strainmix/top10_PT_markers.txt \
        --tool glymph=out/glymph/kidney_nuc2.h5ad --tool cellbender=out/cellbender/kidney_nuc2/kidney_nuc2_filtered.h5 \
        --out out/metrics/kidney_nuc2.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.io import read_counts  # noqa: E402

STRAINS = ("BL6", "CAST", "SvImJ")


def rmsle(est, truth):
    return float(np.sqrt(np.mean((np.log1p(est) - np.log1p(truth)) ** 2))) if len(est) else np.nan


def tau(est, truth):
    return float(kendalltau(est, truth).statistic) if len(est) > 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--replicate", required=True, help="replicate label in the metadata (rep1, nuc2, ...)")
    ap.add_argument("--kind", default="")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--tool", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", help="optional directory for per-cell tables")
    a = ap.parse_args()

    raw = read_counts(a.raw)
    X = sp.csr_matrix(raw.X).astype(np.float64)
    meta = pd.read_csv(a.meta, sep="\t")
    meta = meta[(meta.replicate == a.replicate) & meta.Strain.isin(STRAINS) & meta.contPerCell_binom.notna()]
    meta = meta.drop_duplicates("cell").set_index("cell")
    idx = raw.obs_names.get_indexer(meta.index)
    if (idx < 0).mean() > 0.05:
        raise SystemExit(f"{a.name}: {int((idx < 0).sum())} of {len(idx)} metadata barcodes are not in the raw matrix")
    meta, idx = meta[idx >= 0], idx[idx >= 0]
    names = meta.index.to_numpy()
    truth = meta.contPerCell_binom.to_numpy(float)
    is_cast = (meta.Strain == "CAST").to_numpy()
    is_pt = (meta.celltype == "PT").to_numpy()
    is_nonpt = (meta.celltype.notna() & (meta.celltype != "PT")).to_numpy()

    want = [g.strip() for g in open(a.markers) if g.strip()]
    gname = raw.var["gene_name"].astype(str).to_numpy()
    mk = np.flatnonzero(np.isin(gname, want))
    slc = np.flatnonzero(gname == "Slc34a1")
    if mk.size == 0:
        raise SystemExit(f"{a.name}: none of the marker genes were found by name")
    mark = np.zeros(raw.n_vars)
    mark[mk] = 1.0

    Xe = X[idx]
    rows = []
    for tool, path in [("raw", None)] + [t.split("=", 1) for t in a.tool]:
        if path is None:
            present, Y = np.ones(len(idx), bool), Xe
        else:
            if not os.path.exists(path):
                print(f"[metrics] {a.name}: {tool} output missing ({path}), skipped")
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
        nb, na_ = np.asarray(Xp.sum(1)).ravel(), np.asarray(Y.sum(1)).ravel()
        est = (nb - na_) / np.maximum(nb, 1)
        tr, cast, pt, nonpt = truth[present], is_cast[present], is_pt[present], is_nonpt[present]
        mb, ma = np.asarray(Xp @ mark).ravel(), np.asarray(Y @ mark).ravel()
        diff = (Y - Xp).tocsr()
        det_b = det_a = np.nan
        if slc.size and nonpt.any():
            sb = np.asarray(Xp[nonpt][:, slc].sum(1)).ravel()
            sa = np.asarray(Y[nonpt][:, slc].sum(1)).ravel()
            det_b, det_a = 100 * float((sb >= 0.5).mean()), 100 * float((sa >= 0.5).mean())
        share = lambda m_, n_, sel: (m_[sel].sum() + 1.0) / max(n_[sel].sum(), 1.0)  # noqa: E731
        rows.append({
            "dataset": a.name, "kind": a.kind, "tool": tool,
            "n_eval_cells": int(len(idx)), "n_missing": int(len(idx) - present.sum()),
            "n_cast": int(cast.sum()), "n_pt": int(pt.sum()), "n_nonpt": int(nonpt.sum()), "n_markers": int(mk.size),
            "truth_median": float(np.median(tr)), "est_median": float(np.median(est)),
            "truth_pct": 100 * float((tr * nb).sum() / max(nb.sum(), 1)),
            "removed_pct": 100 * float((nb - na_).sum() / max(nb.sum(), 1)),
            "removed_over_truth": float((nb - na_).sum() / max((tr * nb).sum(), 1e-9)),
            "contam_rmsle": rmsle(est, tr),
            "contam_kendall": tau(est, tr) if tool != "raw" else np.nan,
            "contam_rmsle_cast": rmsle(est[cast], tr[cast]),
            "contam_kendall_cast": tau(est[cast], tr[cast]) if tool != "raw" else np.nan,
            "marker_removed_nonpt": 1.0 - ma[nonpt].sum() / max(mb[nonpt].sum(), 1.0) if nonpt.any() else np.nan,
            "marker_kept_pt": ma[pt].sum() / max(mb[pt].sum(), 1.0) if pt.any() else np.nan,
            "slc34a1_nonpt_before": det_b, "slc34a1_nonpt_after": det_a,
            "marker_log2fc": float(np.log2(share(ma, na_, pt) / share(ma, na_, nonpt))) if pt.any() and nonpt.any() else np.nan,
            "n_above_raw": int((diff.data > 1e-6).sum()),
            "added_counts": float(diff.data[diff.data > 0].sum()),
            "n_negative": int((Y.data < 0).sum()),
            "n_noninteger": int((np.abs(Y.data - np.round(Y.data)) > 1e-6).sum()),
        })
        if a.per_cell and tool != "raw":
            os.makedirs(a.per_cell, exist_ok=True)
            pd.DataFrame({
                "barcode": names[present], "strain": meta.Strain.to_numpy()[present],
                "celltype": meta.celltype.to_numpy()[present], "n_raw": nb, "n_after": na_,
                "truth_frac": tr, "est_frac": est, "marker_before": mb, "marker_after": ma,
            }).to_csv(os.path.join(a.per_cell, f"{a.name}.{tool}.tsv.gz"), sep="\t", index=False)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, sep="\t", index=False, float_format="%.5g")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
