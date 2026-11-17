#!/usr/bin/env python
"""Thresholds from the prior step for many libraries, without fitting anything.

    python diag_thresholds.py name=path [name=path ...]
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import estimate_priors, hazard_dips  # noqa: E402
from glymph.io import read_counts  # noqa: E402

rows = []
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    if not os.path.exists(path):
        continue
    a = read_counts(path)
    X = sp.csr_matrix(a.X)
    N = np.asarray(X.sum(1)).ravel().astype(float)
    d = estimate_priors(X).diagnostics
    dips = [(round(x["T"]), x["n_above"], round(x["lam_knee"] / max(x["lam_above"], 1e-9), 2)) for x in hazard_dips(N)]
    v = ""
    if "vendor_cell" in a.obs:
        vc = a.obs["vendor_cell"].to_numpy()
        v = f" vendor {int(vc.sum())}"
    print(f"{name:22s} {d['threshold_method'][:7]:7s} P {d['empty_level']:6.0f} T {d['cell_threshold']:6.0f} ({d['n_cells_prior']:6d}) "
          f"confident {d['confident_threshold']:6.0f} ({d['n_confident']:6d}){v}  dips (depth, barcodes above, depth of dip) {dips}", flush=True)
    rows.append({"name": name, "barcodes": X.shape[0], "counts": int(N.sum()), "method": d["threshold_method"], "empty_level": round(d["empty_level"]),
                 "cell_threshold": round(d["cell_threshold"]), "confident_threshold": round(d["confident_threshold"]), "confident_cells": d["n_confident"],
                 "vendor_cells": int(a.obs["vendor_cell"].sum()) if "vendor_cell" in a.obs else "",
                 "units": a.obs["unit"].nunique() if "unit" in a.obs else "", "pools": a.obs["pool"].nunique() if "pool" in a.obs else ""})
task = os.environ.get("SLURM_ARRAY_TASK_ID")
if task and rows:
    import pandas as pd

    os.makedirs("out/diag", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"out/diag/thresholds_{task}.tsv", sep="\t", index=False)
