#!/usr/bin/env python
"""Cells with far more silent-gene counts than background explains: their background match and their fate.

For each library of one Glymph run: how many called cells carry a background match (the flagged
ones), how they spread over match bands, what share of their counts was removed in each band, and,
where a species truth table exists, the true background share of the same cells.

    python diag_match.py --tool h3_all --names xeno_HSC_3,bd_hgmm_mix1 [--out out] [--per-cell out/metrics/per_cell]
"""
import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--tool", required=True)
ap.add_argument("--names", required=True)
ap.add_argument("--out", default="out")
ap.add_argument("--per-cell", default="out/metrics/per_cell")
a = ap.parse_args()
edges = [0, 0.3, 0.5, 0.7, 0.9, 10]
for name in a.names.split(","):
    p = os.path.join(a.out, a.tool, name + ".h5ad")
    if not os.path.exists(p):
        continue
    obs = ad.read_h5ad(p, backed="r").obs
    if "glymph_background_match" not in obs:
        print(name, "no match column")
        continue
    m = obs["glymph_background_match"].to_numpy(float)
    fl = ~np.isnan(m)
    rem = obs["glymph_background_fraction"].to_numpy(float)
    raw = obs["glymph_raw_counts"].to_numpy(float)
    spared = obs["glymph_doublet_like"].to_numpy(bool)
    truth = None
    q = os.path.join(a.per_cell, f"{name}.{a.tool}.tsv.gz")
    if os.path.exists(q):
        t = pd.read_csv(q, sep="\t").set_index("barcode")
        truth = t["true_frac" if "true_frac" in t else "truth_frac"].reindex(obs.index).to_numpy(float)
    print(f"== {name} {a.tool}: {len(obs)} cells, {int(fl.sum())} flagged ({100 * fl.mean():.1f}%), {int(spared.sum())} left alone; "
          f"removed share: all {100 * (rem * raw).sum() / raw.sum():.2f}%, unflagged median {100 * np.median(rem[~fl]):.2f}%")
    cm = obs["glymph_cluster_match"].to_numpy(float) if "glymph_cluster_match" in obs else np.full(len(obs), np.nan)
    bands = [(f"match {lo:.1f}-{min(hi, 1.5):.1f}", fl & (m >= lo) & (m < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    if (~np.isnan(cm)).any():
        cut = [0, 0.15, 0.3, 0.5, 0.7, 10]
        bands = [(f"background match >= 0.5, cluster match {lo:.2f}-{min(hi, 1.0):.2f}", fl & (m >= 0.5) & (cm >= lo) & (cm < hi)) for lo, hi in zip(cut[:-1], cut[1:])]
        bands = [("background match < 0.5", fl & ~(m >= 0.5))] + bands
    for label, b in bands:
        if not b.any():
            continue
        t = ""
        if truth is not None and (~np.isnan(truth[b])).any():
            t = f", truth median {100 * np.nanmedian(truth[b]):.1f}% (scored {int((~np.isnan(truth[b])).sum())})"
        print(f"   {label}: {int(b.sum()):5d} cells, median depth {np.median(raw[b]):7.0f}, left alone {int((b & spared).sum()):5d}, "
              f"removed median {100 * np.median(rem[b]):5.1f}%, holds {100 * (rem * raw)[b].sum() / (rem * raw).sum():4.1f}% of removal{t}")
