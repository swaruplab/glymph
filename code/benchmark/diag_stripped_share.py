#!/usr/bin/env python
"""Flagged cells that stay held to their cluster's silent genes: how much of each cell is removed, and is it background?

For one Glymph run and each library: the flagged cells that were not left alone, in bands of the
share of their counts that was removed, with the true background share of the scored ones (species
or genotype truth) and the share of all removal the band holds. A band where far more is removed
than the truth says is a band of cells that carry expression of their own on those genes.

    python diag_stripped_share.py --tool h4_a --names xeno_HSC_3,bd_hgmm_mix1,kidney_rep2
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
edges = [0, 0.10, 0.15, 0.20, 0.30, 2]
for name in a.names.split(","):
    p = os.path.join(a.out, a.tool, name + ".h5ad")
    if not os.path.exists(p):
        continue
    obs = ad.read_h5ad(p, backed="r").obs
    m = obs["glymph_background_match"].to_numpy(float)
    held = ~np.isnan(m) & ~obs["glymph_doublet_like"].to_numpy(bool)
    rem = obs["glymph_background_fraction"].to_numpy(float)
    raw = obs["glymph_raw_counts"].to_numpy(float)
    q = os.path.join(a.per_cell, f"{name}.{a.tool}.tsv.gz")
    truth = np.full(len(obs), np.nan)
    if os.path.exists(q):
        t = pd.read_csv(q, sep="\t").set_index("barcode")
        truth = t["true_frac" if "true_frac" in t else "truth_frac"].reindex(obs.index).to_numpy(float)
    sc = ~np.isnan(truth)
    print(f"== {name} {a.tool}: {int(held.sum())} flagged cells held to the silent genes ({int((held & sc).sum())} scored); scored cells in all {int(sc.sum())}, "
          f"removed there {100 * (rem * raw)[sc].sum() / raw[sc].sum():.2f}%, truth {100 * (truth * raw)[sc].sum() / raw[sc].sum():.2f}%")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = held & (rem >= lo) & (rem < hi)
        if not b.any():
            continue
        bs = b & sc
        extra = f"; scored {int(bs.sum())}: removed {100 * (rem * raw)[bs].sum() / max(raw[bs].sum(), 1):.1f}% of their counts, truth {100 * (truth * raw)[bs].sum() / max(raw[bs].sum(), 1):.1f}%, " \
                f"{100 * (rem * raw)[bs].sum() / max((rem * raw)[sc].sum(), 1):.1f}% of all removal in scored cells" if bs.any() else ""
        print(f"   removed {100 * lo:.0f}-{100 * min(hi, 1):.0f}%: {int(b.sum()):5d} cells, median depth {np.median(raw[b]):7.0f}{extra}")
