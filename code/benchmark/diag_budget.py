#!/usr/bin/env python
"""Why does Glymph under-remove? Compare its background budget with the truth, per library.

For evaluation cells of one species, cross-species counts are pure background. If background is
ambient (constant per droplet) plus swapping (proportional to depth), then
    cross = eps * e * soup_share_other + s * N * pool_share_other
so a straight-line fit of cross counts on N gives the true ambient level (intercept) and the
true swap fraction (slope) to set beside what Glymph learned (e and s in its report).

    python diag_budget.py --run-dir . --tool glymph
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=".")
ap.add_argument("--tool", default="glymph")
ap.add_argument("--out", default=None)
a = ap.parse_args()

met = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(os.path.join(a.run_dir, "out/metrics/*.tsv")))])
rows = []
for f in sorted(glob.glob(os.path.join(a.run_dir, f"out/metrics/per_cell/*.{a.tool}.tsv.gz"))):
    name = os.path.basename(f).split(".")[0]
    pc = pd.read_csv(f, sep="\t")
    rep_path = os.path.join(a.run_dir, f"out/{a.tool}/{name}.report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    e = float(list(rep.get("e_unit", {"all": np.nan}).values())[0])
    s = float(list(rep.get("swap_fraction", {"all": np.nan}).values())[0])
    h_share = float(met[(met.dataset == name) & (met.tool == "raw")]["ambient_human_share"].iloc[0])
    tot = pc.groupby("species")["n_raw"].sum()
    pool_h = tot.get("human", 0) / tot.sum()  # share of cell UMIs that are human: the pool-average profile split
    for sp_name, other_soup, other_pool in (("human", 1 - h_share, 1 - pool_h), ("mouse", h_share, pool_h)):
        d = pc[pc.species == sp_name]
        if len(d) < 50:
            continue
        b, a0 = np.polyfit(d.n_raw, d.cross_before, 1)
        rows.append({
            "library": name, "cells_of": sp_name, "n": len(d), "median_N": int(d.n_raw.median()),
            "cross_frac_pct": round(100 * d.cross_before.sum() / d.n_raw.sum(), 2),
            "fit_intercept": round(a0, 1), "fit_slope_pct": round(100 * b, 3),
            "true_ambient_per_cell": round(a0 / other_soup, 0), "glymph_e": round(e, 0),
            "true_swap_pct": round(100 * b / other_pool, 2), "glymph_s_pct": round(100 * s, 2),
            "soup_share_other": round(other_soup, 2), "pool_share_other": round(other_pool, 2),
            "removed_pct": round(100 * (1 - d.cross_after.sum() / d.cross_before.sum()), 1),
            "corr_cross_N": round(np.corrcoef(d.n_raw, d.cross_before)[0, 1], 2),
        })
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
if a.out:
    df.to_csv(a.out, sep="\t", index=False)
