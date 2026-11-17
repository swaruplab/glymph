#!/usr/bin/env python
"""Development helper: print the kidney and species scores of chosen tool folders side by side.

    python show_dev_scores.py out/metrics dev3b dev3c
"""
import glob
import os
import sys

import pandas as pd

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
folder, tools = sys.argv[1], sys.argv[2:]
t = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(os.path.join(folder, "*.tsv")))], ignore_index=True)
k = t[t.dataset.str.startswith("kidney") & t.tool.isin(["raw"] + tools)]
if len(k):
    cols = ["dataset", "tool", "truth_pct", "removed_pct", "removed_over_truth", "contam_rmsle", "contam_kendall", "contam_rmsle_cast",
            "contam_kendall_cast", "marker_removed_nonpt", "marker_kept_pt", "slc34a1_nonpt_after", "marker_log2fc"]
    print(k[[c for c in cols if c in k]].round(3).to_string(index=False))
    pub = {"kidney_rep1": (0.084, 0.44), "kidney_rep2": (0.085, 0.43), "kidney_rep3": (0.052, 0.37), "kidney_nuc2": (0.113, 0.32), "kidney_nuc3": (0.168, 0.10)}
    print("published CellBender 0.2.0 (Janssen 2023), RMSLE and Kendall:", pub)
s = t[~t.dataset.str.startswith("kidney") & t.tool.isin(tools)]
if len(s):
    cols = ["dataset", "tool", "n_eval_cells", "raw_cross_species_pct", "cross_species_removed", "own_retained", "native_loss",
            "same_species_ratio", "contam_rmsle", "contam_kendall", "n_above_raw"]
    print(s[[c for c in cols if c in s]].sort_values(["dataset", "tool"]).round(4).to_string(index=False))
    print(s.groupby("tool")[["cross_species_removed", "own_retained", "native_loss", "contam_rmsle", "contam_kendall"]].mean().round(4).to_string())
