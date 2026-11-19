"""Spliced and unspliced molecules as a second negative control (nuclei).

Ambient RNA in a nuclei preparation is mostly cytoplasmic: mature, spliced. A nucleus's own RNA
is largely nascent: it still carries introns. So whether a molecule is unspliced says where it
came from, independently of how many molecules of the gene the cell has.

Measured on the mouse kidney nuclei of Janssen et al. 2023 (2026-11-18), for ten proximal-tubule
marker genes: 4% of their molecules are unspliced in the empty droplets, 10% in the nuclei of
other cell types (where they can only be background), and 40% in proximal-tubule nuclei. As one
number per cell the unspliced share is a weak measure of the background fraction (correlation
with the genotype truth 0.31, because nuclei also keep some of their own cytoplasm), so it is used
per gene:

  pi_amb[g]   unspliced share of gene g in the empty partitions (measured)
  pi_bg[g]    the same for background inside cells: pi_amb, moved by a fixed shift on the logit
              scale (0 by default: the same RNA inside and outside cells)
  pi_nat[g]   unspliced share of the gene's own expression, from the cluster that expresses it
              most, corrected for the background that cluster carries

A cluster's molecules of a gene then say how much of them is background:

  support   the unspliced count is significantly below what 20% of own expression would give:
            at least four fifths background. The gene is silent there even when the count test
            cannot say so. In a dirty library that is the common case for the genes that matter
            most: tubule markers in other kidney nuclei sit at 46% of their level in tubule
            nuclei, far above the "a fifth of the top cluster" rule for a candidate.
  veto      the unspliced count is significantly above what 25% of own expression would give:
            the cluster expresses the gene, however few its counts. It is not silent.

and each entry's background share is updated by its own molecules (splice_update), so that the
counts taken from a gene come from the cells whose molecules of it look like background.

Input: layers["unspliced"] beside X (velocyto, STARsolo Velocyto, alevin-fry, or
code/benchmark/51_bam_splice_counts.py from a Cell Ranger alignment file). Without it nothing here runs.
"""

from __future__ import annotations

import torch

F64 = torch.float64


@torch.no_grad()
def sums_of(csr, values: torch.Tensor, rows: torch.Tensor, group: torch.Tensor, n_groups: int, max_entries: int = 30_000_000):
    """Like DeviceCSR.group_sums, for a second value per entry (aligned with csr.data): float64 [n_groups, n_genes]."""
    G = csr.shape[1]
    out = torch.zeros(n_groups * G, dtype=F64, device=csr.device)
    for a, b in csr.row_chunks(rows, max_entries):
        r, c, _, pos = csr.take(rows[a:b])
        out.index_add_(0, group[a:b][r] * G + c, values[pos].to(F64))
    return out.view(n_groups, G)


def _logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


@torch.no_grad()
def splice_profiles(Xe, Ue, Xc, Uc, Bc, tested, big, level: float, prior: float = 10.0, min_top: float = 20.0, min_gap: float = 0.1,
                    shift: float = 0.0):
    """Per-gene unspliced shares of background and of own expression.

    Xe, Ue [G]: molecules and unspliced molecules of each gene in the empty partitions.
    Xc, Uc, Bc [k, G]: the same per cluster, and the fitted background counts. tested [k, G]: the
    genes the count test calls silent per cluster. big [k]: clusters with enough cells.
    Returns a dict with pi_amb, pi_bg, pi_nat, informative (bool [G]) and diagnostics.
    """
    Xe, Ue = Xe.to(F64), Ue.to(F64)
    pe = float(Ue.sum() / Xe.sum().clamp_min(1.0))
    pi_amb = (Ue + prior * pe) / (Xe + prior)
    # one logit shift from the empty partitions to background inside cells, on the tested silent genes
    t = tested.to(F64)
    xs, us = (Xc * t).sum(0), (Uc * t).sum(0)
    la = _logit(pi_amb)
    n_sil = float(xs.sum())
    fitted = None
    if n_sil >= 200.0:
        grid = torch.linspace(-2.0, 3.0, 101, dtype=F64, device=Xc.device)
        p = torch.sigmoid(la.unsqueeze(0) + grid.unsqueeze(1)).clamp(1e-6, 1 - 1e-6)
        ll = (us.unsqueeze(0) * torch.log(p) + (xs - us).unsqueeze(0) * torch.log1p(-p)).sum(1)
        fitted = float(grid[int(ll.argmax())])
    # The shift that the count-silent genes ask for is reported, not used (shift < 0 uses it). On the kidney nuclei
    # it came out at 1.3 to 3.0, a background a third to a half unspliced, where tubule markers in other cell types
    # show a tenth: genes that are silent by their counts still hold nascent transcripts of the cell's own, and a
    # background profile fitted on them takes that expression for background (2026-11-18).
    shift = (fitted or 0.0) if shift < 0 else float(shift)
    pi_bg = torch.sigmoid(la + shift)
    # own expression: the cluster with the highest rate of the gene, less the background it carries
    rate = Xc / Xc.sum(1, keepdim=True).clamp_min(1.0)
    top = torch.where(big.unsqueeze(1), rate, torch.full_like(rate, -1.0)).argmax(0)
    g = torch.arange(Xc.shape[1], device=Xc.device)
    xt, ut, bt = Xc[top, g], Uc[top, g], Bc[top, g]
    f = (level * bt / xt.clamp_min(1.0)).clamp(0.0, 0.9)
    pi_top = ut / xt.clamp_min(1.0)
    pi_nat = ((pi_top - f * pi_bg) / (1.0 - f)).clamp(0.0, 0.999)
    pi_nat = torch.maximum(pi_nat, pi_bg)
    inform = (xt >= min_top) & (pi_nat - pi_bg >= min_gap)
    q = lambda v: [round(float(x), 3) for x in torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], dtype=F64, device=v.device))]  # noqa: E731
    diag = {"unspliced_share_empty": round(pe, 4), "unspliced_share_cells": round(float(Uc.sum() / Xc.sum().clamp_min(1.0)), 4),
            "background_logit_shift": round(shift, 3), "shift_the_count_silent_genes_ask_for": None if fitted is None else round(fitted, 3),
            "silent_molecules_for_shift": int(n_sil), "informative_genes": int(inform.sum())}
    if int(inform.sum()) > 10:
        diag["pi_bg_q10_q50_q90"], diag["pi_nat_q10_q50_q90"] = q(pi_bg[inform]), q(pi_nat[inform])
    return {"pi_amb": pi_amb, "pi_bg": pi_bg, "pi_nat": pi_nat, "informative": inform, "top": top, "diag": diag}


def splice_support(n, u, pi_bg, pi_nat, inform, own: float = 0.2, tail: float = 0.01, min_n: float = 5.0):
    """n molecules with u unspliced are at least 1 - own background: P(Poisson(n * share at `own` own expression) <= u) < tail."""
    thr = pi_bg + own * (pi_nat - pi_bg)
    lam = (n * thr).clamp_min(1e-12)
    return inform & (n >= min_n) & (torch.special.gammaincc(u + 1.0, lam) < tail)


def splice_veto(n, u, pi_bg, pi_nat, inform, own: float = 0.25, tail: float = 0.01, min_u: float = 3.0):
    """n molecules with u unspliced hold more than `own` own expression: P(Poisson(n * share at `own`) >= u) < tail."""
    thr = pi_bg + own * (pi_nat - pi_bg)
    lam = (n * thr).clamp_min(1e-12)
    return inform & (u >= min_u) & (torch.special.gammainc(u.clamp_min(1e-6), lam) < tail)


def splice_update(share, v, u, pi_bg, pi_nat, inform):
    """Background share of an entry after looking at its molecules.

    share: the model's background share of the entry (v molecules, u of them unspliced; pi_bg,
    pi_nat, inform already gathered per entry). Each molecule is background with probability
    share * P(its splicing | background) / (that + (1 - share) * P(its splicing | own)); the
    entry's new share is the mean over its molecules. Entries of uninformative genes are unchanged.
    """
    u = torch.minimum(u.to(share.dtype), v)
    pb, pn = pi_bg.to(share.dtype).clamp(1e-3, 1 - 1e-3), pi_nat.to(share.dtype).clamp(1e-3, 1 - 1e-3)
    s = share.clamp(1e-6, 1 - 1e-6)
    p_un = s * pb / (s * pb + (1 - s) * pn)
    p_ex = s * (1 - pb) / (s * (1 - pb) + (1 - s) * (1 - pn))
    new = (u * p_un + (v - u) * p_ex) / v.clamp_min(1.0)
    return torch.where(inform, new, share)
