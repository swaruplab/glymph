#!/usr/bin/env python
"""Species-mixing (human + mouse) metrics for ambient removal tools.

The metrics follow the published benchmarks so numbers can be set beside theirs:

  cross_species_removed    share of known cross-species counts removed, aggregate
                           ("sensitivity" in Schroeder et al. 2026)
  median_cell_removal      median over cells of the same share (the per-cell statistic
                           quoted in the AmbiDose preprint)
  own_retained             share of same-species counts kept
  same_species_ratio       same-species counts removed / expected same-species
                           contamination, where the expectation extrapolates each cell's
                           cross-species counts by the species split of the ambient pool
                           (ideal = 1; above 1 means native counts were removed)
  native_loss              estimated share of native counts removed in excess of that
                           expectation (over-correction)
  n_above_raw, added_counts, n_negative, n_noninteger
                           count integrity, the criterion Schroeder et al. argue must be
                           primary: a decontaminated entry must not exceed the raw entry
  contam_rmsle, contam_kendall
                           per-cell contamination fraction, estimated vs truth
                           (the per-cell accuracy endpoints of Janssen 2023 and Nie 2026)

Truth: in a barcode assigned to one species, every UMI from the other species is
background. Same-species background is invisible and is extrapolated, as in Schroeder
et al. Evaluation cells are chosen from the raw matrix alone, so every tool is scored
on the same barcodes; barcodes a tool did not call are reported as missing.

Same-species truth, measured (columns ending in _measured, added 2026-11-18). The extrapolation above
assumes the background inside a cell has the species split of the empty partitions. It can be checked where
the background is visible, human RNA in mouse cells against mouse RNA in human cells, and it does not always
hold: in hgmm_10k_v3 the empty partitions are 88% human and the background inside cells is 70% human, so the
extrapolation expects three times too much same-species background in human cells and a third of it in
mouse cells; in jurkat_32d it is 1.75 and 0.54 times. The measured version takes, for a cell of one species
and its depth, the ratio of (the background of its own species that cells of the other species receive at
that depth) to (the background of the other species that cells of its species receive), each a constant
plus a share of depth fitted on the cleanest nine tenths of the cells, and multiplies the cell's
cross-species counts by it. It assumes only that the background does not depend on the species of the
cell it lands in. Both versions are reported; no tool's output enters either.

Alignment artifacts. A few genes collect reads from the wrong genome (ribosomal RNA reads
on mouse Gm42418 and Lars2 are the known case), so their "cross-species" counts are not RNA
from another cell and no ambient remover should be scored on them. A gene is masked when,
in the raw data alone, it holds at least 0.5% of the cross-species counts of one species'
cells and that share is more than ten times its share both in the other species' own cells
and in the other-species part of the low-count barcodes. Masked genes are left out of every
score for every tool; --no-mask-artifacts gives the unmasked scores, and the columns
n_masked_genes and masked_cross_pct say how much was set aside.

    python species_mix_metrics.py --name hgmm_10k_v3 --raw raw.h5 \
        --tool glymph=out/glymph/hgmm_10k_v3.h5ad --tool cellbender=out/cb/x_filtered.h5 \
        --out metrics.tsv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.background import ordmag_threshold  # noqa: E402
from glymph.io import read_counts  # noqa: E402


def fit_line(n: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y ~ Poisson(a + b * n) with a, b >= 0 (EM updates): background counts as a constant plus a share of depth."""
    if n.size < 20 or y.sum() <= 0:
        return np.array([np.nan, np.nan])
    th = np.array([max(y.mean() * 0.5, 1e-6), max(y.mean() * 0.5 / max(n.mean(), 1.0), 1e-12)])
    Z = np.stack([np.ones_like(n, dtype=np.float64), n.astype(np.float64)], 1)
    col = Z.sum(0)
    for _ in range(5000):
        mu = np.maximum(Z @ th, 1e-12)
        new = th * ((y / mu)[:, None] * Z).sum(0) / col
        if np.all(np.abs(new - th) <= 1e-7 * np.maximum(th, 1e-12)):
            th = new
            break
        th = new
    return th


def gene_species(var: pd.DataFrame) -> np.ndarray:
    """'human', 'mouse' or '' per gene, from the genome column or the gene id prefix."""
    lab = var["genome"].astype(str).to_numpy() if "genome" in var else np.array([""] * len(var))
    if len({x for x in lab if x}) < 2:
        lab = var["gene_id"].astype(str).str.split("_").str[0].to_numpy()
    out = np.array([""] * len(lab), dtype=object)
    low = np.char.lower(lab.astype(str))
    out[np.char.startswith(low, "grch") | np.char.startswith(low, "hg")] = "human"
    out[np.char.startswith(low, "grcm") | np.char.startswith(low, "mm")] = "mouse"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--tool", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", help="optional directory for per-cell tables")
    ap.add_argument("--purity", type=float, default=0.9, help="min share of one species to call a singlet")
    ap.add_argument("--cells", help="evaluate on these barcodes (one per line) instead of the order-of-magnitude threshold; "
                                    "for platforms where that threshold does not apply")
    ap.add_argument("--no-mask-artifacts", action="store_true", help="score every gene, alignment artifacts included")
    ap.add_argument("--artifact-min-share", type=float, default=0.005)
    ap.add_argument("--artifact-fold", type=float, default=10.0)
    ap.add_argument("--artifacts-out", help="optional table of the masked genes")
    args = ap.parse_args()

    raw = read_counts(args.raw)
    X = sp.csr_matrix(raw.X)
    sp_gene = gene_species(raw.var)
    hm = np.stack([(sp_gene == "human"), (sp_gene == "mouse")]).astype(np.float64)  # [2, G]
    if hm.sum(1).min() == 0:
        raise SystemExit(f"{args.name}: could not find both human and mouse genes")
    per_sp = np.asarray(X @ hm.T)  # [n, 2] UMIs per species
    N = per_sp.sum(1)

    T, _ = ordmag_threshold(np.sort(N)[::-1])
    purity = per_sp.max(1) / np.maximum(N, 1)
    above = N >= T
    if args.cells:
        listed = raw.obs_names.isin([x.strip() for x in open(args.cells) if x.strip()])
        above = listed
        T = float(np.quantile(N[listed], 0.01)) if listed.any() else T  # the background window below stays under the cells
    is_eval = above & (purity >= args.purity)
    own = per_sp.argmax(1)  # 0 human, 1 mouse
    bg = (N >= 5) & (N <= T / 10.0)
    amb_split = per_sp[bg].sum(0) / per_sp[bg].sum()  # species split of the ambient pool
    n_multiplet = int((above & (purity < args.purity)).sum())

    ev = np.flatnonzero(is_eval)
    own_e = own[ev]

    # Alignment artifacts, from the raw data alone (see the module docstring).
    n_masked, masked_pct, art_rows = 0, 0.0, []
    if not args.no_mask_artifacts:
        gname = raw.var["gene_name"].astype(str).to_numpy() if "gene_name" in raw.var else raw.var_names.astype(str).to_numpy()
        soup_g = np.asarray(X[bg].sum(0)).ravel().astype(np.float64)
        keep_gene = np.ones(raw.n_vars, bool)
        cross_all = 0.0
        for o in (0, 1):  # cells of species o, genes of species 1 - o
            other = hm[1 - o] > 0
            cells_o, cells_x = ev[own_e == o], ev[own_e == 1 - o]
            if cells_o.size == 0 or cells_x.size == 0:
                continue
            cross = np.asarray(X[cells_o].sum(0)).ravel() * other
            src = np.asarray(X[cells_x].sum(0)).ravel() * other
            soup_o = soup_g * other
            cs = cross / max(cross.sum(), 1.0)
            ref = np.maximum(src / max(src.sum(), 1.0), soup_o / max(soup_o.sum(), 1.0))
            flag = (cs >= args.artifact_min_share) & (cs > args.artifact_fold * ref)
            cross_all += cross.sum()
            masked_pct += cross[flag].sum()
            keep_gene &= ~flag
            for g in np.flatnonzero(flag):
                art_rows.append({"dataset": args.name, "gene": gname[g], "in_cells_of": ("human", "mouse")[o],
                                 "share_of_cross_pct": 100 * cs[g], "share_in_source_cells_pct": 100 * src[g] / max(src.sum(), 1.0),
                                 "share_in_soup_pct": 100 * soup_o[g] / max(soup_o.sum(), 1.0)})
        n_masked = int((~keep_gene).sum())
        masked_pct = 100 * masked_pct / max(cross_all, 1.0)
        if n_masked:
            print(f"[metrics] {args.name}: masked {n_masked} alignment-artifact genes holding {masked_pct:.1f}% of cross-species counts: "
                  + ", ".join(f"{r['gene']} ({r['share_of_cross_pct']:.1f}% in {r['in_cells_of']} cells)" for r in art_rows))
            hm = hm * keep_gene[None, :]
            per_sp = np.asarray(X @ hm.T)
            amb_split = per_sp[bg].sum(0) / per_sp[bg].sum()
            # species, purity and the singlet call again, now without the masked genes
            purity = per_sp.max(1) / np.maximum(per_sp.sum(1), 1)
            is_eval = above & (purity >= args.purity)
            own = per_sp.argmax(1)
            n_multiplet = int((above & (purity < args.purity)).sum())
            ev = np.flatnonzero(is_eval)
            own_e = own[ev]
        if args.artifacts_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.artifacts_out)), exist_ok=True)
            pd.DataFrame(art_rows, columns=["dataset", "gene", "in_cells_of", "share_of_cross_pct", "share_in_source_cells_pct",
                                            "share_in_soup_pct"]).to_csv(args.artifacts_out, sep="\t", index=False, float_format="%.4g")

    # Background of each species as the cells of the other species receive it: a constant plus a share of depth,
    # on the cleanest nine tenths of those cells (see "Same-species truth, measured" in the module docstring).
    N_ev = per_sp[ev].sum(1)
    line = {}
    for o in (0, 1):  # line[o]: background of species 1 - o in cells of species o
        c = own_e == o
        n_, y_ = N_ev[c], per_sp[ev][c, 1 - o]
        if c.sum() >= 20:
            sh = y_ / np.maximum(n_, 1)
            k_ = sh <= np.quantile(sh, 0.9)
            line[o] = fit_line(n_[k_], y_[k_])
        else:
            line[o] = np.array([np.nan, np.nan])
    measured_ok = bool(np.all(np.isfinite(line[0])) and np.all(np.isfinite(line[1])))

    ev_names = raw.obs_names[ev]
    Xe = X[ev]
    rows = []
    tools = [("raw", None)] + [t.split("=", 1) for t in args.tool]
    for tool, path in tools:
        if path is None:
            present, Yp = np.ones(ev.size, bool), Xe.astype(np.float64)
        else:
            if not os.path.exists(path):
                print(f"[metrics] {args.name}: {tool} output missing ({path}), skipped")
                continue
            a = read_counts(path, integer=False)
            # Put the tool's genes in raw gene order (genes it dropped become zeros).
            gi = pd.Index(raw.var["gene_id"].astype(str)).get_indexer(a.var["gene_id"].astype(str))
            ok = gi >= 0
            proj = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), gi[ok])), shape=(a.n_vars, raw.n_vars))
            A = sp.csr_matrix(a.X).astype(np.float64) @ proj
            ri = a.obs_names.get_indexer(ev_names)
            present = ri >= 0
            Yp = sp.csr_matrix(A[ri[present]])

        Xp, own_p = Xe[present].astype(np.float64), own_e[present]
        n = int(present.sum())
        before = np.asarray(Xp @ hm.T)
        after = np.asarray(Yp @ hm.T)
        idx = np.arange(n)
        own_b, own_a = before[idx, own_p], after[idx, own_p]
        cr_b, cr_a = before[idx, 1 - own_p], after[idx, 1 - own_p]
        ratio = amb_split[own_p] / np.maximum(amb_split[1 - own_p], 1e-9)
        exp_own = np.minimum(cr_b * ratio, own_b)  # expected same-species background
        own_removed = own_b - own_a
        diff = (Yp - Xp).tocsr()
        with np.errstate(divide="ignore", invalid="ignore"):
            cell_rm = 1.0 - cr_a[cr_b > 0] / cr_b[cr_b > 0]
        Nb = before.sum(1)
        true_frac = (cr_b + exp_own) / np.maximum(Nb, 1)
        est_frac = (Nb - after.sum(1)) / np.maximum(Nb, 1)
        if measured_ok:
            la, lb = np.stack([line[0], line[1]])[own_p].T  # the other species' background in cells of this species
            oa, ob = np.stack([line[0], line[1]])[1 - own_p].T  # this species' background in cells of the other species
            ratio_m = (oa + ob * Nb) / np.maximum(la + lb * Nb, 1e-9)
            exp_own_m = np.minimum(cr_b * ratio_m, own_b)
            true_frac_m = (cr_b + exp_own_m) / np.maximum(Nb, 1)
        else:
            exp_own_m = np.full_like(exp_own, np.nan)
            true_frac_m = np.full_like(true_frac, np.nan)
        row = {
            "dataset": args.name,
            "tool": tool,
            "n_eval_cells": int(ev.size),
            "n_missing": int(ev.size - n),
            "n_multiplets_excluded": n_multiplet,
            "raw_cross_species_pct": 100 * cr_b.sum() / max(Nb.sum(), 1),
            "cross_species_removed": 1.0 - cr_a.sum() / max(cr_b.sum(), 1),
            "median_cell_removal": float(np.median(cell_rm)) if cell_rm.size else np.nan,
            "own_retained": own_a.sum() / max(own_b.sum(), 1),
            "same_species_ratio": own_removed.sum() / max(exp_own.sum(), 1e-9),
            "native_loss": max(0.0, own_removed.sum() - exp_own.sum()) / max(own_b.sum() - exp_own.sum(), 1),
            "n_above_raw": int((diff.data > 1e-6).sum()),
            "added_counts": float(diff.data[diff.data > 0].sum()),
            "n_negative": int((Yp.data < 0).sum()),
            "n_noninteger": int((np.abs(Yp.data - np.round(Yp.data)) > 1e-6).sum()),
            "contam_rmsle": float(np.sqrt(np.mean((np.log1p(est_frac) - np.log1p(true_frac)) ** 2))),
            "contam_kendall": float(kendalltau(est_frac, true_frac).statistic) if tool != "raw" else np.nan,
            "same_species_ratio_measured": own_removed.sum() / max(np.nansum(exp_own_m), 1e-9) if measured_ok else np.nan,
            "native_loss_measured": max(0.0, own_removed.sum() - np.nansum(exp_own_m)) / max(own_b.sum() - np.nansum(exp_own_m), 1) if measured_ok else np.nan,
            "contam_rmsle_measured": float(np.sqrt(np.mean((np.log1p(est_frac) - np.log1p(true_frac_m)) ** 2))) if measured_ok else np.nan,
            "contam_kendall_measured": float(kendalltau(est_frac, true_frac_m).statistic) if (measured_ok and tool != "raw") else np.nan,
            "own_expected_measured_over_extrapolated": float(np.nansum(exp_own_m) / max(exp_own.sum(), 1e-9)) if measured_ok else np.nan,
            "ambient_human_share": float(amb_split[0]),
            "cell_threshold": float(T),
            "n_masked_genes": n_masked,
            "masked_cross_pct": masked_pct,
        }
        rows.append(row)
        if args.per_cell and tool != "raw":
            os.makedirs(args.per_cell, exist_ok=True)
            pd.DataFrame(
                {"barcode": ev_names[present], "species": np.where(own_p == 0, "human", "mouse"), "n_raw": Nb,
                 "cross_before": cr_b, "cross_after": cr_a, "own_before": own_b, "own_after": own_a,
                 "true_frac": true_frac, "est_frac": est_frac, "true_frac_measured": true_frac_m}
            ).to_csv(os.path.join(args.per_cell, f"{args.name}.{tool}.tsv.gz"), sep="\t", index=False)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.5g")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
