#!/usr/bin/env python
"""For each library: the plateau candidate, how homogeneous its barcodes are, and the knee.

Written to choose the rule that separates a true empty-partition plateau (10x) from a cell
population mistaken for one (BD Rhapsody tables, split-pool data), on every library at hand.

    python diag_prior_modes.py out.tsv name=path [name=path ...]
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import hazard_knee, ordmag_threshold, peak_dispersion, plateau_mixture  # noqa: E402
from glymph.io import read_counts  # noqa: E402

rows = []
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    try:
        a = read_counts(path)
        X = sp.csr_matrix(a.X)
        N = np.asarray(X.sum(1)).ravel().astype(float)
        mix = plateau_mixture(N)
        knee = hazard_knee(N)
        om, n_om = ordmag_threshold(np.sort(N)[::-1])
        r = {"name": name, "barcodes": X.shape[0], "ordmag_T": round(om), "ordmag_n": n_om}
        if mix:
            lo, hi = mix["P"] * 10 ** (-mix["sd0"]), mix["P"] * 10 ** (mix["sd0"])
            r.update({"P": round(mix["P"]), "sd0": round(mix["sd0"], 3), "plateau_T": round(mix["T"]), "n_empty": round(mix["n_empty"]),
                      "n_cells": round(mix["n_cells"]), "disp_peak": round(peak_dispersion(X, N, lo, hi), 2),
                      "disp_peak_200": round(peak_dispersion(X, N, lo, hi, n_genes=200), 2)})
            top = N[N >= mix["T"]]
            if top.size >= 100:
                q = np.quantile(top, [0.25, 0.75])
                r["disp_above"] = round(peak_dispersion(X, N, q[0], q[1]), 2)
        if knee:
            r.update({"knee_T": round(knee["T"]), "knee_hi": round(knee["hi"]), "knee_n": knee["n_above"],
                      "lam_knee": round(knee["lam_knee"], 4), "lam_above": round(knee["lam_above"], 4), "lam_below": round(knee["lam_below"], 4)})
            sub = N[(N >= 5) & (N < knee["hi"])]
            if sub.size:
                o = np.argsort(sub)
                c = np.cumsum(sub[o])
                r["bg_umi_median"] = round(float(sub[o][np.searchsorted(c, 0.5 * c[-1])]))
                r["bg_n"] = int(sub.size)
                r["disp_tail"] = round(peak_dispersion(X, N, max(5, knee["hi"] / 4), knee["hi"]), 2)
        if "vendor_cell" in a.obs:
            v = a.obs["vendor_cell"].to_numpy()
            r["vendor_n"] = int(v.sum())
            r["vendor_min"] = round(float(np.quantile(N[v], 0.01))) if v.any() else ""
    except Exception as e:
        r = {"name": name, "error": f"{type(e).__name__}: {e}"[:120]}
    rows.append(r)
    print(r, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], sep="\t", index=False)
