#!/usr/bin/env python
"""One line per ingested library from the <name>.ingest.json files: sizes, thresholds, matches.

    python summarize_ingest.py <std folder> [config/datasets.tsv] > ingest_summary.tsv
"""
import glob
import json
import os
import sys

import pandas as pd

std = sys.argv[1]
cfg = pd.read_csv(sys.argv[2], sep="\t", dtype=str).fillna("") if len(sys.argv) > 2 else None
rows = []
for f in sorted(glob.glob(os.path.join(std, "*.ingest.json"))):
    d = json.load(open(f))
    p = d.get("priors", {})
    rows.append({
        "name": d["name"], "barcodes": d["n_barcodes"], "genes": d["n_genes"], "counts_M": round(d["n_counts"] / 1e6, 1),
        "q99_depth": round(d["depth_quantiles"]["0.99"]), "method": p.get("threshold_method", d.get("priors_error", ""))[:22],
        "empty_level": round(p.get("empty_level", 0)), "cell_thr": round(p.get("cell_threshold", 0)),
        "cells_above": int(p.get("n_cells_prior", 0)), "vendor_cells": d.get("vendor_cells_matched", ""),
        "vendor_above_thr": p.get("vendor_cells_above_threshold", ""), "units": d.get("n_unit", ""), "pools": d.get("n_pool", ""),
        "labels": d.get("labels_matched", d.get("wells_with_tissue", "")),
        "genomes": ",".join(f"{k}:{v}" for k, v in list(d.get("genomes", {}).items())[:3]),
    })
t = pd.DataFrame(rows)
if cfg is not None:
    t = cfg[["name", "tier", "platform", "kind", "truth"]].merge(t, on="name", how="left")
t.to_csv(sys.stdout, sep="\t", index=False)
