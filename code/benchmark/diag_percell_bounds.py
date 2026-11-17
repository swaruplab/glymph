#!/usr/bin/env python
"""Kidney per-cell scores of simple reference estimates, from a metrics per-cell table.

How well could any estimate do that uses depth alone (a constant plus a slope, fitted to the truth
itself), and how does the tool's estimate relate to depth? Separates "wrong amount per cell" from
"truth is noisy" when a per-cell error or rank correlation looks poor.

    python diag_percell_bounds.py out/metrics/per_cell/kidney_nuc2.dev3e.tsv.gz [more tables]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

rm = lambda a, b: float(np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2)))  # noqa: E731
tau = lambda a, b: float(kendalltau(a, b).statistic)  # noqa: E731
for f in sys.argv[1:]:
    t = pd.read_csv(f, sep="\t")
    N, tr, est = t.n_raw.to_numpy(float), t.truth_frac.to_numpy(float), t.est_frac.to_numpy(float)
    b = tr * N
    A = np.vstack([np.ones_like(N), N]).T
    (c0, c1), *_ = np.linalg.lstsq(A, b, rcond=None)
    plane = np.clip((c0 + c1 * N) / N, 0, 1)
    flat = np.clip(np.median(b) / N, 0, 1)
    const = np.full_like(N, np.median(tr))
    print(f"{f.split('/')[-1]}: {len(t)} cells, truth fraction median {np.median(tr):.3f}, estimate median {np.median(est):.3f}")
    print(f"   tool estimate               RMSLE {rm(est, tr):.3f}  Kendall {tau(est, tr):.3f}   (estimate vs 1/depth: Kendall {tau(est, 1 / N):.3f})")
    print(f"   truth-fitted line in depth  RMSLE {rm(plane, tr):.3f}  Kendall {tau(plane, tr):.3f}   (line {c0:.0f} + {c1:.4f} N)")
    print(f"   same counts in every cell   RMSLE {rm(flat, tr):.3f}  Kendall {tau(flat, tr):.3f}")
    print(f"   same fraction in every cell RMSLE {rm(const, tr):.3f}")
    for lab, sel in (("PT", t.celltype == "PT"), ("non-PT", (t.celltype != "PT") & t.celltype.notna()), ("CAST", t.strain == "CAST")):
        s_ = sel.to_numpy()
        if s_.sum() > 50:
            print(f"   {lab:7s} n {s_.sum():6d}: truth median {np.median(tr[s_]):.3f}, estimate median {np.median(est[s_]):.3f}, RMSLE {rm(est[s_], tr[s_]):.3f}, Kendall {tau(est[s_], tr[s_]):.3f}")
