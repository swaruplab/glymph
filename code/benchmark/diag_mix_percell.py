#!/usr/bin/env python
"""Where do the cross-species counts of a mixture sit, and which of them does each tool remove?

Reads the per-cell tables written by species_mix_metrics.py (--per-cell). Prints, for one library:
  the spread of the per-cell cross-species share and how much of all cross-species counts
  each band of cells holds, with every tool's removal inside that band;
  a straight-line fit of cross-species counts on depth, per species (flat = ambient per
  partition, sloped = background that grows with depth);
  for split-pool libraries (barcode ends in a well label), the species make-up of each well
  and the cross-species share of its cells.

    python diag_mix_percell.py --dir out/metrics/per_cell --name parse_hgmm_sub2 --tools glymph,cellbender,scar
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tools", default="glymph,cellbender,scar")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")

    tabs = {}
    for t in a.tools.split(","):
        p = os.path.join(a.dir, f"{a.name}.{t}.tsv.gz")
        if os.path.exists(p):
            tabs[t] = pd.read_csv(p, sep="\t").set_index("barcode")
    base = next(iter(tabs.values()))
    frac = base.cross_before / base.n_raw.clip(lower=1)
    print(f"== {a.name}: {len(base)} cells, cross-species share of all counts {100 * base.cross_before.sum() / base.n_raw.sum():.3f}%")
    print("per-cell cross share quantiles (5, 25, 50, 75, 95, 99):",
          np.round(100 * np.quantile(frac, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]), 3), "%")
    print("depth quantiles (5, 50, 95):", np.quantile(base.n_raw, [0.05, 0.5, 0.95]).astype(int))

    edges = [0, 0.005, 0.01, 0.02, 0.05, 0.101]
    band = pd.cut(frac, edges, include_lowest=True)
    rows = []
    for b, idx in base.groupby(band, observed=True).groups.items():
        r = {"cross share band": str(b), "cells": len(idx), "pct of cross counts": 100 * base.loc[idx, "cross_before"].sum() / base.cross_before.sum()}
        for t, d in tabs.items():
            k = d.index.intersection(idx)
            r[f"{t} removed"] = 1 - d.loc[k, "cross_after"].sum() / max(d.loc[k, "cross_before"].sum(), 1)
            r[f"{t} own kept"] = d.loc[k, "own_after"].sum() / max(d.loc[k, "own_before"].sum(), 1)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\ncross counts = a + b * depth, per species")
    for s, d in base.groupby("species"):
        A = np.stack([np.ones(len(d)), d.n_raw.to_numpy(float)], 1)
        coef, *_ = np.linalg.lstsq(A, d.cross_before.to_numpy(float), rcond=None)
        med = np.median(d.n_raw)
        print(f"  {s:6s} n={len(d):6d}  a={coef[0]:8.2f}  b={coef[1]:.5f}  at median depth {med:.0f}: flat part {coef[0]:.1f}, depth part {coef[1] * med:.1f}"
              f"  | mean cross {d.cross_before.mean():.1f}, median {d.cross_before.median():.1f}")

    well = base.index.to_series().str.rsplit("_", n=1).str[-1]
    if well.map(lambda w: bool(re.fullmatch(r"[A-H]\d{1,2}", str(w)))).mean() > 0.9:
        g = pd.DataFrame({"well": well, "species": base.species, "frac": frac, "n": base.n_raw, "cross": base.cross_before})
        t = g.groupby("well").agg(cells=("n", "size"), human_share=("species", lambda x: (x == "human").mean()),
                                  median_depth=("n", "median"), cross_pct=("cross", "sum"))
        t["cross_pct"] = 100 * t.cross_pct / g.groupby("well").n.sum()
        t["median_cell_cross_pct"] = 100 * g.groupby("well").frac.median()
        print("\nper well (first round barcode):")
        print(t.to_string())
        mixed = t[(t.human_share > 0.1) & (t.human_share < 0.9)]
        pure = t[(t.human_share <= 0.1) | (t.human_share >= 0.9)]
        print(f"\nwells with both species: {len(mixed)} (cross {mixed.cross_pct.mean():.2f}%), single-species wells: {len(pure)} (cross {pure.cross_pct.mean():.2f}%)")


if __name__ == "__main__":
    main()
