#!/usr/bin/env python
"""Collect the tier 2 (mouse kidney strain mixture) tables and draw the summary figure.

Reads   <run>/out/metrics/*.tsv, <run>/out/<tool>/*.time.txt, Glymph reports
Writes  kidney_metrics.tsv, runtime.tsv, kidney_summary.pdf into --out
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from collect_results import dot_rows_figure, runtime_table


def kidney_figure(met, rt, path):
    d = met[met.tool != "raw"].copy()
    if not rt.empty:
        d = d.merge(rt[["dataset", "tool", "wall_s"]], on=["dataset", "tool"], how="left")
    else:
        d["wall_s"] = float("nan")
    d["marker_removed"] = 100 * d.marker_removed_nonpt
    d["marker_kept"] = 100 * d.marker_kept_pt
    panels = [
        ("contam_rmsle", "Per-cell contamination error", "against genotype truth; left is better", "{:.3f}", False),
        ("contam_kendall", "Per-cell rank agreement", "Kendall tau; right is better", "{:.2f}", False),
        ("marker_removed", "PT markers removed from non-PT cells, %", "background; right is better", "{:.1f}", False),
        ("marker_kept", "PT markers kept in PT cells, %", "real signal; right is better", "{:.1f}", False),
        ("wall_s", "Wall time per library, s", "left is better, log scale", "{:.0f}", True),
    ]
    dot_rows_figure(d, panels, path, "Three cell and two nuclei libraries (Janssen et al. 2023).", sort_col="contam_rmsle",
                    ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.run_dir, "out/metrics/*.tsv")))
    met = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True) if files else pd.DataFrame()
    met.to_csv(os.path.join(a.out, "kidney_metrics.tsv"), sep="\t", index=False)
    rt = runtime_table(a.run_dir, met)
    rt.to_csv(os.path.join(a.out, "runtime.tsv"), sep="\t", index=False)
    if met.empty:
        print("no metrics found")
        return
    kidney_figure(met, rt, os.path.join(a.out, "kidney_summary.pdf"))
    cols = ["dataset", "tool", "n_eval_cells", "n_missing", "truth_pct", "removed_pct", "contam_rmsle", "contam_kendall",
            "marker_removed_nonpt", "marker_kept_pt", "slc34a1_nonpt_before", "slc34a1_nonpt_after", "n_above_raw", "n_noninteger"]
    print(met[cols].to_string(index=False))


if __name__ == "__main__":
    main()
