#!/usr/bin/env python
"""Cells that lose far more than their true background, per library and tool (species mixtures).

A cell counts when more than 10% of its counts were removed and that is more than three times its
true background share plus 3 points. Reads the per-cell tables of species_mix_metrics.py; where they carry the
truth with the species split measured inside cells, the count is given against that truth too.

    python count_stripped_cells.py out/metrics/per_cell out/deliverables/stripped_cells.tsv [tool ...]
"""
import glob
import os
import sys

import pandas as pd

src, out, tools = sys.argv[1], sys.argv[2], set(sys.argv[3:])
rows = []
for f in sorted(glob.glob(os.path.join(src, "*.tsv.gz"))):
    name, tool = os.path.basename(f)[:-7].rsplit(".", 1)
    if tools and tool not in tools:
        continue
    t = pd.read_csv(f, sep="\t")
    if "true_frac" not in t.columns:
        continue
    rem = t.n_raw - t.own_after - t.cross_after
    hit = (t.est_frac > 0.10) & (t.est_frac > 3 * t.true_frac + 0.03)
    rows.append({"dataset": name, "tool": tool, "cells": len(t), "cells_stripped_far_beyond_truth": int(hit.sum()),
                 "of_them_minor_species": int((hit & (t.species == t.species.value_counts().idxmin())).sum()) if t.species.nunique() > 1 else 0,
                 "their_share_of_removal_pct": round(100 * rem[hit].sum() / max(rem.sum(), 1), 1),
                 "removed_pct": round(100 * rem.sum() / t.n_raw.sum(), 2), "truth_pct": round(100 * (t.true_frac * t.n_raw).sum() / t.n_raw.sum(), 2)})
    if "true_frac_measured" in t.columns and t.true_frac_measured.notna().any():  # the same count against the truth with the split measured inside cells
        hit_m = (t.est_frac > 0.10) & (t.est_frac > 3 * t.true_frac_measured + 0.03)
        rows[-1].update({"cells_stripped_far_beyond_measured_truth": int(hit_m.sum()),
                         "truth_measured_pct": round(100 * (t.true_frac_measured * t.n_raw).sum() / t.n_raw.sum(), 2)})
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
print(f"{len(rows)} rows written to {out}")
