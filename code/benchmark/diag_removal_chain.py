#!/usr/bin/env python
"""Kidney: where does removal fall short of the truth? Follows the background amount down the
pipeline for the cells with a genotype truth: truth counts, the multiplier the model ended with
(epsilon x e), the counts the posterior attributes to background, and the counts removed.

    python diag_removal_chain.py <glymph out.h5ad> <report.json> <replicate> <cell_metadata.tsv.gz>
"""
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

out = ad.read_h5ad(sys.argv[1], backed="r")
rep = json.load(open(sys.argv[2]))
m = pd.read_csv(sys.argv[4], sep="\t")
m = m[(m.replicate == sys.argv[3]) & m.contPerCell_binom.notna()].drop_duplicates("cell").set_index("cell")
o = out.obs.reindex(m.index).dropna(subset=["glymph_raw_counts"])
m = m.loc[o.index]
e = float(list(rep["e_unit"].values())[0])
N = o.glymph_raw_counts.to_numpy(float)
b = m.contPerCell_binom.to_numpy(float) * N
amb_model = o.glymph_epsilon.to_numpy(float) * e
exp_bg = (o.glymph_expected_ambient + o.glymph_expected_swapped).to_numpy(float)
rem = o.glymph_removed_counts.to_numpy(float)
c = rep["swap_calibration"]
p0 = c["pools"]["0"]
print(f"{sys.argv[3]}: {len(o)} truth cells; e {e:.0f}; calibration kappa {p0.get('kappa'):.3f}, depth term {p0.get('kappa_depth'):.3f} at reference depth {p0.get('depth_ref'):.0f}, s {p0.get('s'):.4f}")
trend = (p0["kappa"] + p0["kappa_depth"] * N / p0["depth_ref"]) * e + p0["s"] * N
q = np.quantile(N, np.linspace(0, 1, 6))
print("   depth bin        cells   truth  trend(calib)  eps*e(model)  expected bg  removed   removed/truth")
for i in range(5):
    s_ = (N >= q[i]) & (N <= q[i + 1])
    print(f"   {q[i]:6.0f}-{q[i + 1]:6.0f}  {s_.sum():6d}  {b[s_].mean():6.0f}  {trend[s_].mean():10.0f}  {amb_model[s_].mean():11.0f}  {exp_bg[s_].mean():10.0f}  {rem[s_].mean():7.0f}   {rem[s_].sum() / b[s_].sum():.2f}")
print(f"   all                      {b.mean():6.0f}  {trend.mean():10.0f}  {amb_model.mean():11.0f}  {exp_bg.mean():10.0f}  {rem.mean():7.0f}   {rem.sum() / b.sum():.2f}")
