#!/usr/bin/env python
"""Which cells lose far more counts than their true background, and which genes do they lose?

For one species mixture and one tool: cells whose removed share is above 10% and above three
times the truth plus 3 points are listed with the genes that lost the most counts, the share of
the loss on the top genes, and the tool's doublet-like flag when the output has one.

    python diag_heavy_cells.py --raw data/std/xeno_HSC_3.h5ad --out out/h1_a/xeno_HSC_3.h5ad \
        --per-cell out/metrics/per_cell/xeno_HSC_3.h1_a.tsv.gz [--top 8]
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-cell", required=True)
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()
pc = pd.read_csv(a.per_cell, sep="\t").set_index("barcode")
heavy = pc[(pc.est_frac > 0.10) & (pc.est_frac > 3 * pc.true_frac + 0.03)].sort_values("est_frac", ascending=False)
out = ad.read_h5ad(a.out)
raw = ad.read_h5ad(a.raw)[out.obs_names, out.var_names]
flag = out.obs["glymph_doublet_like"] if "glymph_doublet_like" in out.obs else pd.Series(False, index=out.obs_names)
print(f"{len(pc)} cells scored, {len(heavy)} heavy; flagged doublet-like in the output: {int(flag.sum())} of {out.n_obs}; heavy and flagged: {int(flag.reindex(heavy.index).fillna(False).sum())}")
fl = flag.reindex(pc.index).fillna(False).astype(bool)
rem = pc.n_raw - pc.own_after - pc.cross_after
print(f"flagged cells among scored: {int(fl.sum())}, hold {100 * pc.n_raw[fl].sum() / pc.n_raw.sum():.1f}% of counts, {100 * rem[fl].sum() / rem.sum():.1f}% of removal; "
      f"removed share median flagged {np.median((rem / pc.n_raw)[fl]) if fl.any() else float('nan'):.3f}, others {np.median((rem / pc.n_raw)[~fl]):.3f}; "
      f"true share median flagged {np.median(pc.true_frac[fl]) if fl.any() else float('nan'):.3f}, others {np.median(pc.true_frac[~fl]):.3f}")
R, O = sp.csr_matrix(raw.X), sp.csr_matrix(out.X)
idx = {b: i for i, b in enumerate(out.obs_names)}
agg = np.zeros(out.n_vars)
for b in heavy.index[: a.top * 4]:
    i = idx[b]
    d = np.asarray((R[i] - O[i]).todense()).ravel()
    agg += d
for b in heavy.index[: a.top]:
    i = idx[b]
    d = np.asarray((R[i] - O[i]).todense()).ravel()
    r = np.asarray(R[i].todense()).ravel()
    top = np.argsort(-d)[:10]
    print(f"{b} {pc.species[b]} raw {int(r.sum())} removed {int(d.sum())} ({d.sum() / r.sum():.2f}; truth {pc.true_frac[b]:.3f}) flagged {bool(flag[b])} genes losing counts {int((d > 0).sum())} top10 share {d[top].sum() / max(d.sum(), 1):.2f}: "
          + ", ".join(f"{out.var_names[g]}({int(d[g])}/{int(r[g])})" for g in top))
top = np.argsort(-agg)[:30]
print("genes losing the most over the heavy cells: " + ", ".join(f"{out.var_names[g]}({int(agg[g])})" for g in top))
