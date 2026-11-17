#!/usr/bin/env python
"""What are the cross-species counts of a mixture made of?

For the cells of one species, the counts on the other species' genes are compared with three
candidate sources:
  cells      the average profile of the other species' cells (what barcode swapping would bring)
  soup       the other-species part of the low-count barcodes (what ambient RNA would bring)
  ortholog   the cell's own expression moved to the gene of the same name in the other genome
             (what reads aligned to the wrong genome would bring)
A profile concentrated on a few genes that follow the ortholog column points to alignment,
not to RNA from other cells.

    python diag_cross_profile.py --raw data/std/parse_hgmm_sub2.h5ad \
        --per-cell out/metrics/per_cell/parse_hgmm_sub2.glymph.tsv.gz [--pure-wells]
"""

from __future__ import annotations

import argparse
import re

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def cos(a, b):
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--per-cell", required=True)
    ap.add_argument("--pure-wells", action="store_true", help="use only cells from wells that hold one species (split-pool)")
    a = ap.parse_args()

    A = ad.read_h5ad(a.raw)
    X = sp.csr_matrix(A.X)
    name = A.var["gene_name"].astype(str).to_numpy() if "gene_name" in A.var else A.var_names.astype(str).to_numpy()
    gid = A.var["gene_id"].astype(str).to_numpy() if "gene_id" in A.var else A.var_names.astype(str).to_numpy()
    genome = A.var["genome"].astype(str).to_numpy() if "genome" in A.var else np.array([g.split("_")[0] for g in gid])
    low = np.char.lower(genome.astype(str))
    is_h = np.char.startswith(low, "grch") | np.char.startswith(low, "hg")
    is_m = np.char.startswith(low, "grcm") | np.char.startswith(low, "mm")
    sym = np.array([re.sub(r"^(GRCh38|GRCm38|GRCm39|mm10|hg19|hg38)_+", "", n) for n in name])
    print(f"{a.raw}: {A.n_obs} barcodes, {int(is_h.sum())} human genes, {int(is_m.sum())} mouse genes; example names {name[:2]} {name[-2:]}")

    pc = pd.read_csv(a.per_cell, sep="\t").set_index("barcode")
    if a.pure_wells:
        well = pc.index.to_series().str.rsplit("_", n=1).str[-1]
        share = (pc.species == "human").groupby(well).mean()
        pure = share[(share == 0) | (share == 1)].index
        pc = pc[well.isin(pure).to_numpy()]
        print(f"single-species wells: {sorted(pure)}; {len(pc)} cells kept")
    ix = A.obs_names.get_indexer(pc.index)
    N = np.asarray(X.sum(1)).ravel()
    T = np.quantile(N[ix], 0.01)
    bg = (N >= 5) & (N <= T / 10)
    soup = np.asarray(X[bg].sum(0)).ravel()

    up = pd.Series(np.arange(len(sym)), index=np.char.upper(sym.astype(str)))
    for own, other, own_mask, other_mask in (("human", "mouse", is_h, is_m), ("mouse", "human", is_m, is_h)):
        cells_own = ix[(pc.species == own).to_numpy()]
        cells_other = ix[(pc.species == other).to_numpy()]
        cross = np.asarray(X[cells_own].sum(0)).ravel() * other_mask          # other-species genes in own-species cells
        own_prof = np.asarray(X[cells_own].sum(0)).ravel() * own_mask
        other_cells = np.asarray(X[cells_other].sum(0)).ravel() * other_mask
        soup_o = soup * other_mask
        # ortholog projection: own expression of the gene with the same (upper-case) symbol
        oi = np.flatnonzero(other_mask)
        own_idx = pd.Series(np.flatnonzero(own_mask), index=np.char.upper(sym[own_mask].astype(str)))
        own_idx = own_idx[~own_idx.index.duplicated()]
        match = own_idx.reindex(np.char.upper(sym[oi].astype(str))).to_numpy()
        orth = np.zeros(len(sym))
        ok = ~np.isnan(match)
        orth[oi[ok]] = own_prof[match[ok].astype(int)]
        tot = cross.sum()
        order = np.argsort(-cross)
        print(f"\n== {other} genes in {own} cells: {tot:.0f} counts ({100 * tot / max(own_prof.sum() + tot, 1):.2f}% of the cells' counts), {len(cells_own)} cells")
        print(f"   concentration: top 10 genes {100 * cross[order[:10]].sum() / tot:.1f}%, top 100 {100 * cross[order[:100]].sum() / tot:.1f}%"
              f"  | same for the {other} cells' own profile: {100 * np.sort(other_cells)[::-1][:10].sum() / max(other_cells.sum(), 1):.1f}%, {100 * np.sort(other_cells)[::-1][:100].sum() / max(other_cells.sum(), 1):.1f}%")
        print(f"   cosine with: {other} cells {cos(cross, other_cells):.3f} | soup {cos(cross, soup_o):.3f} | ortholog of own expression {cos(cross, orth):.3f}"
              f"  (soup vs {other} cells {cos(soup_o, other_cells):.3f})")
        # non-negative least squares mix of the three sources
        from scipy.optimize import nnls
        S = np.stack([other_cells / max(other_cells.sum(), 1), soup_o / max(soup_o.sum(), 1), orth / max(orth.sum(), 1)], 1)
        w, _ = nnls(S[oi], cross[oi] / tot)
        print(f"   least-squares mix: cells {w[0]:.2f}, soup {w[1]:.2f}, ortholog {w[2]:.2f}")
        t = pd.DataFrame({"gene": sym[order[:20]], "cross_pct": 100 * cross[order[:20]] / tot,
                          f"{other}_cells_pct": 100 * other_cells[order[:20]] / max(other_cells.sum(), 1),
                          "soup_pct": 100 * soup_o[order[:20]] / max(soup_o.sum(), 1),
                          "ortholog_pct": 100 * orth[order[:20]] / max(orth.sum(), 1)})
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
