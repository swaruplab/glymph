#!/usr/bin/env python
"""One line per Glymph experiment: mean over libraries of the main scores and learned globals."""

import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default=".")
ap.add_argument("--metrics-dir", default=None)
a = ap.parse_args()
mdir = a.metrics_dir or os.path.join(a.run_dir, "out/metrics_exp")
met = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(os.path.join(mdir, "*.tsv")))])
met = met[met.tool != "raw"]
rows = []
for tool, d in met.groupby("tool"):
    s, ep, tr, wall = [], [], [], []
    for name in d.dataset:
        rp = os.path.join(a.run_dir, f"out/{tool}/{name}.report.json")
        if os.path.exists(rp):
            r = json.load(open(rp))
            s.append(list(r["swap_fraction"].values())[0])
            ep.append(r["train"]["epochs_run"])
            tr.append(r["train"]["steps"])
            wall.append(r["timings"]["total_s"])
    rows.append({
        "tool": tool, "n": len(d),
        "removed_mean": round(100 * d.cross_species_removed.mean(), 1),
        "removed_min": round(100 * d.cross_species_removed.min(), 1),
        "native_loss_pct": round(100 * d.native_loss.mean(), 3),
        "own_kept_pct": round(100 * d.own_retained.mean(), 2),
        "same_ratio": round(d.same_species_ratio.mean(), 2),
        "rmsle": round(d.contam_rmsle.mean(), 4),
        "kendall": round(d.contam_kendall.mean(), 3),
        "s_pct": round(100 * sum(s) / len(s), 2) if s else None,
        "steps": int(sum(tr) / len(tr)) if tr else None,
        "total_s": round(sum(wall) / len(wall), 0) if wall else None,
    })
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).sort_values("removed_mean", ascending=False).to_string(index=False))
wide = met.pivot(index="tool", columns="dataset", values="cross_species_removed").mul(100).round(1)
print()
print(wide.to_string())
