#!/usr/bin/env python
"""Development helper: one row per variant, means over the kidney libraries, the seven tuning
mixtures and the held-out mixtures, counting only libraries every listed variant has finished.

    python compare_variants.py out/metrics dev3e dev3f dev3g
"""
import glob
import os
import sys

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
folder, tools = sys.argv[1], sys.argv[2:]
t = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(os.path.join(folder, "*.tsv")))], ignore_index=True)
t = t[t.tool.isin(tools)]
TUNE = ["hgmm_1k_v3", "hgmm_5k_v3", "hgmm_6k_v2", "hgmm_10k_v3", "hgmm_12k_v2", "hgmm_20k_v31_HT", "hgmm_10k_gemx_v4"]
groups = {
    "kidney cells (rep1-3)": (t.dataset.isin(["kidney_rep1", "kidney_rep2", "kidney_rep3"]), ["contam_rmsle", "contam_kendall", "removed_over_truth", "marker_removed_nonpt", "marker_kept_pt"]),
    "kidney nuclei (nuc2-3)": (t.dataset.isin(["kidney_nuc2", "kidney_nuc3"]), ["contam_rmsle", "contam_kendall", "removed_over_truth", "marker_removed_nonpt", "marker_kept_pt"]),
    "tuning mixtures (7)": (t.dataset.isin(TUNE), ["cross_species_removed", "own_retained", "native_loss", "same_species_ratio", "contam_rmsle", "contam_kendall"]),
    "held-out mixtures": (~t.dataset.isin(TUNE) & ~t.dataset.str.startswith("kidney"), ["cross_species_removed", "own_retained", "native_loss", "same_species_ratio", "contam_rmsle", "contam_kendall"]),
}
for name, (sel, cols) in groups.items():
    g = t[sel]
    if g.empty:
        continue
    n_tools = g.groupby("dataset").tool.nunique()
    full = n_tools[n_tools == len(set(g.tool))].index
    g = g[g.dataset.isin(full)]
    print(f"\n{name}: {len(full)} libraries finished by all of {sorted(set(g.tool))}: {', '.join(sorted(full))}")
    if len(full):
        print(g.groupby("tool")[[c for c in cols if c in g]].mean().round(4).to_string())
