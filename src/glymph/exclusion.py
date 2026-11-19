"""A label-free estimate of the swapped fraction: the exclusion bound.

Why it is needed. The likelihood cannot pin the swapped fraction s from below: if s is too
small, the decoder puts a little native mass on the genes a cell does not express and the
fit is just as good. On real human + mouse mixtures Glymph 0.0.1 learned s 2 to 2.5 times
too low, and the background it failed to account for stayed in the output.

The estimate. Take a group c of similar cells and a gene g. The group's count total is
native + ambient + swapped, and native cannot be negative, so

    u_cg = (X_cg - A_cg) / D_cg  >=  s
    X_cg  observed counts of gene g summed over the group
    A_cg  expected ambient counts, sum over the group's cells of e_unit * soup_unit[g]
    D_cg  swapped counts per unit s, sum over the group's cells of N * chibar_pool[g]

with near-equality for genes the group does not express. Across genes, u is about 1 where the
group expresses the gene at the pool's average level and piles up at s where the gene is
silent. So for each group the lowest peak of the histogram of log10(u), searched below a cap,
estimates s, and the median over groups is the estimate for the pool. Groups come from
k-means on the model's own latent space. Nothing here uses species or any other label.

On seven 10x human + mouse mixtures the estimate landed within about 25% of the truth read
from cross-species counts (for example 1.8% for a true 1.4 to 1.7%, where the model alone had
learned 0.8%). It has not been checked on tissue. It returns None when fewer than three
groups show a peak, which is what a single homogeneous population should produce; the caller
then keeps the learned value.
"""

from __future__ import annotations

import math
import os

import torch


def kmeans(z: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """Plain k-means with k-means++ seeding. Returns a label per row."""
    n = z.shape[0]
    gen = torch.Generator().manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=gen))
    centers = [z[first]]
    d2 = ((z - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        w = d2.clamp_min(0).cpu().double()
        j = int(torch.multinomial(w / w.sum(), 1, generator=gen)) if float(w.sum()) > 0 else int(torch.randint(n, (1,), generator=gen))
        centers.append(z[j])
        d2 = torch.minimum(d2, ((z - z[j]) ** 2).sum(1))
    C = torch.stack(centers).clone()
    lab = torch.zeros(n, dtype=torch.long, device=z.device)
    for _ in range(iters):
        lab = torch.cdist(z, C).argmin(1)
        sums = torch.zeros_like(C).index_add_(0, lab, z)
        cnt = torch.bincount(lab, minlength=k).to(z.dtype).unsqueeze(1)
        C = torch.where(cnt > 0, sums / cnt.clamp_min(1), C)
    return lab


def _lowest_peak(logu: torch.Tensor, cap: float, bin_width: float = 0.1, min_count: float = 5.0, lo: float = -4.0, hi: float = 2.0):
    """Peak of the smoothed histogram of log10(u) below log10(cap); None if it sits at the cap."""
    nb = int(round((hi - lo) / bin_width))
    h = torch.histc(logu.clamp(lo, hi - 1e-6).float().cpu(), bins=nb, min=lo, max=hi)
    sm = h.clone()
    sm[1:-1] = 0.25 * h[:-2] + 0.5 * h[1:-1] + 0.25 * h[2:]
    n_low = int(math.floor((math.log10(cap) - lo) / bin_width + 1e-9))  # bins that lie wholly below the cap
    if n_low < 3:
        return None
    k = int(torch.argmax(sm[:n_low]))
    if k >= n_low - 1 or float(sm[k]) < min_count:
        return None  # still rising at the cap, or too few genes: no silent-gene peak
    a, b = lo + (k - 1) * bin_width, lo + (k + 2) * bin_width
    near = logu[(logu >= a) & (logu < b)]
    return float(10.0 ** near.mean()) if near.numel() else None


@torch.no_grad()
def exclusion_swap_fraction(
    csr,
    rows: torch.Tensor,
    z: torch.Tensor,
    unit: torch.Tensor,
    pool: torch.Tensor,
    e: torch.Tensor,
    soup: torch.Tensor,
    chibar: torch.Tensor,
    n_clusters: int = 12,
    min_cells: int = 20,
    min_expected: float = 30.0,
    cap: float = 0.1,
    seed: int = 0,
    quantile: float = 0.5,
):
    """Estimate s for every pool. rows: confident cells (indices into csr); z, unit, pool: per row.

    quantile: which point of the per-group estimates to take. The bound holds for every group,
    and a group that mixes cell kinds reads high, so a value below the median is the safer read.

    Returns (list with one float or None per pool, diagnostics dict).
    """
    n_pools, G = chibar.shape
    U = soup.shape[0]
    out, diag = [], {}
    amb_profile = e.unsqueeze(1) * soup  # [U, G] expected ambient counts per cell of each unit
    for p in range(n_pools):
        sel = torch.nonzero(pool == p, as_tuple=True)[0]
        k = int(min(n_clusters, sel.numel() // 50))
        if k < 2:
            out.append(None)
            diag[str(p)] = {"reason": "too few confident cells", "n_cells": int(sel.numel())}
            continue
        lab = kmeans(z[sel], k, seed=seed)
        X = csr.group_sums(rows[sel], lab, k)
        N_c = X.sum(1)
        n_cu = torch.zeros(k * U, device=X.device, dtype=torch.float64)
        n_cu.index_add_(0, lab * U + unit[sel], torch.ones(sel.numel(), device=X.device, dtype=torch.float64))
        A = n_cu.view(k, U) @ amb_profile.double()
        D = N_c.unsqueeze(1) * chibar[p].double().unsqueeze(0)
        sizes = torch.bincount(lab, minlength=k)
        per = []
        for ci in range(k):
            if int(sizes[ci]) < min_cells:
                continue
            ok = (0.01 * D[ci] >= min_expected)
            u = (X[ci, ok] - A[ci, ok]) / D[ci, ok]
            u = u[u > 1e-4]
            if u.numel() < 50:
                continue
            est = _lowest_peak(torch.log10(u), cap)
            if est is not None:
                per.append(est)
        per.sort()
        if len(per) >= 3:
            pos = quantile * (len(per) - 1)
            i0 = int(math.floor(pos))
            i1 = min(i0 + 1, len(per) - 1)
            s_hat = per[i0] + (pos - i0) * (per[i1] - per[i0])
            out.append(float(min(s_hat, cap)))
        else:
            out.append(None)
        diag[str(p)] = {"n_cells": int(sel.numel()), "n_clusters": k, "cluster_estimates": [round(x, 5) for x in per]}
    return out, diag


@torch.no_grad()
def exclusion_budget(
    csr,
    rows: torch.Tensor,
    z: torch.Tensor,
    depth: torch.Tensor,
    unit: torch.Tensor,
    pool: torch.Tensor,
    e: torch.Tensor,
    soup: torch.Tensor,
    chibar: torch.Tensor,
    n_clusters: int = 12,
    n_strata: int = 3,
    min_cells: int = 30,
    min_expected: float = 30.0,
    max_fraction: float = 0.8,
    s_max: float = 0.1,
    seed: int = 0,
):
    """Label-free background budget per cell: an ambient multiplier and a swapped fraction.

    Added in 0.3.0 after the first tissue libraries. There the background is mostly ambient
    (about one plateau's worth of counts per cell, whatever the cell's depth), while in
    cell-line mixtures most of it grows with depth. One number cannot describe both, and the
    model alone lets both drift low. So this reads two numbers off the data:

        background per cell = kappa * e * soup  +  s * N * chibar

    For a group of similar cells and a gene the group does not express, the group's counts
    are all background, so in units of "expected ambient counts at kappa = 1"

        m = X_cg / A_cg  =  kappa + s * R,      R = D_cg / A_cg

    with A the expected ambient counts and D the swapped counts per unit s. Groups are k-means
    clusters of the latent space, each split into depth strata, so R differs severalfold
    between groups of the same kind of cell. For each group m is the lowest peak of the
    histogram of X/A over genes (as in exclusion_swap_fraction) and R is the median of D/A
    over the genes at that peak. A robust line through the (R, m) points gives kappa
    (intercept) and s (slope). This is the same regression that recovers the truth from
    cross-species counts in a barnyard, done without knowing the species.

    Returns (kappa per pool, s per pool, diagnostics). An entry is None when too few groups
    show a silent-gene peak; s alone is None when the groups do not span enough depth to
    separate the two, and then kappa carries the whole budget at the caller's current s.
    """
    n_pools, G = chibar.shape
    U = soup.shape[0]
    amb_profile = (e.unsqueeze(1) * soup).double()
    kappas, swaps, diag = [], [], {}
    for p in range(n_pools):
        sel = torch.nonzero(pool == p, as_tuple=True)[0]
        k = int(min(n_clusters, sel.numel() // (2 * min_cells * n_strata)))
        if k < 2:
            kappas.append(None)
            swaps.append(None)
            diag[str(p)] = {"reason": "too few confident cells", "n_cells": int(sel.numel())}
            continue
        lab = kmeans(z[sel], k, seed=seed)
        d = depth[sel].double()
        strat = torch.zeros(sel.numel(), dtype=torch.long, device=lab.device)
        for c in range(k):
            m_ = lab == c
            if int(m_.sum()) >= n_strata * min_cells:
                q = torch.quantile(d[m_], torch.linspace(0, 1, n_strata + 1, device=d.device, dtype=d.dtype)[1:-1])
                strat[m_] = torch.bucketize(d[m_], q)
        gid = lab * n_strata + strat
        ng = k * n_strata
        X = csr.group_sums(rows[sel], gid, ng)
        n_gu = torch.zeros(ng * U, device=X.device, dtype=torch.float64)
        n_gu.index_add_(0, gid * U + unit[sel], torch.ones(sel.numel(), device=X.device, dtype=torch.float64))
        A = n_gu.view(ng, U) @ amb_profile
        Ntot = torch.zeros(ng, device=X.device, dtype=torch.float64).index_add_(0, gid, d)
        D = Ntot.unsqueeze(1) * chibar[p].double().unsqueeze(0)
        sizes = torch.bincount(gid, minlength=ng)
        ms, Rs = [], []
        for g in range(ng):
            if int(sizes[g]) < min_cells:
                continue
            ok = A[g] >= min_expected
            if int(ok.sum()) < 50:
                continue
            u = X[g, ok] / A[g, ok]
            lu = torch.log10(u.clamp_min(1e-12))
            pos = u > 1e-3
            if int(pos.sum()) < 50:
                continue
            cap = max_fraction * float(Ntot[g] / A[g].sum())  # background is at most this share of the group's counts
            est = _lowest_peak(lu[pos], cap, lo=-3.0, hi=4.0)
            if est is None:
                continue
            near = (lu - math.log10(est)).abs() < 0.15
            ratio = D[g, ok] / A[g, ok]
            ms.append(est)
            Rs.append(float(torch.median(ratio[near])) if bool(near.any()) else float(Ntot[g] / A[g].sum()))
        info = {"n_cells": int(sel.numel()), "n_groups": ng, "n_estimates": len(ms),
                "points": [[round(a_, 2), round(b_, 3)] for a_, b_ in zip(Rs, ms)]}
        if len(ms) < 3:
            kappas.append(None)
            swaps.append(None)
            diag[str(p)] = info
            continue
        m_t = torch.tensor(ms, dtype=torch.float64)
        R_t = torch.tensor(Rs, dtype=torch.float64)
        s_hat = None
        if len(ms) >= 6 and float(R_t.max()) >= 1.5 * float(R_t.min()):
            i, j = torch.triu_indices(len(ms), len(ms), offset=1)
            dR = R_t[j] - R_t[i]
            keep = dR.abs() > 0.1 * float(R_t.mean())
            if int(keep.sum()) >= 5:
                s_hat = float(torch.median((m_t[j] - m_t[i])[keep] / dR[keep]).clamp(0.0, s_max))  # Theil-Sen slope
        if s_hat is not None:
            kappa = float(torch.median(m_t - s_hat * R_t))
            if kappa < 0.25:  # nearly all of it grows with depth: keep a floor on ambient and refit the slope
                kappa = 0.25
                s_hat = float(torch.median((m_t - kappa) / R_t).clamp(0.0, s_max))
        else:
            kappa = None  # the caller fits kappa at its own s
        info["kappa"], info["s"] = kappa, s_hat
        info["m_median"], info["R_median"] = float(m_t.median()), float(R_t.median())
        kappas.append(None if kappa is None else float(min(kappa, 4.0)))
        swaps.append(s_hat)
        diag[str(p)] = info
    return kappas, swaps, diag


def _leftmost_peak(logu: torch.Tensor, bin_width: float = 0.1, lo: float = -2.0, hi: float = 2.0, frac: float = 0.25, min_count: float = 20.0):
    """Lowest peak of the smoothed histogram of log10(u) that reaches frac of the tallest one.

    Real expression can only lift a gene above the background level, never push it below, so the
    silent genes are the lowest group of the distribution, not necessarily the largest.
    Returns (level, histogram) with level None when no such peak exists.
    """
    nb = int(round((hi - lo) / bin_width))
    h = torch.histc(logu.clamp(lo, hi - 1e-6).float().cpu(), bins=nb, min=lo, max=hi)
    sm = h.clone()
    sm[1:-1] = 0.25 * h[:-2] + 0.5 * h[1:-1] + 0.25 * h[2:]
    need = max(frac * float(sm[1:].max()), min_count)
    for k in range(1, nb - 1):
        if sm[k] >= sm[k - 1] and sm[k] > sm[k + 1] and float(sm[k]) >= need:
            a, b = lo + (k - 1) * bin_width, lo + (k + 2) * bin_width
            near = logu[(logu >= a) & (logu < b)]
            return (float(10.0 ** near.mean()) if near.numel() else None), h
    return None, h


def _poisson_plane(y, Z, w, theta, lo, hi, iters: int = 500, shrink=None):
    """Weighted Poisson fit of y ~ Z @ theta with boxed non-negative coefficients (EM updates).

    shrink: optional per-coefficient penalty as a share of that column's total (an exponential
    prior). A few percent barely moves a coefficient the data pin down and decides between two
    coefficients the data cannot tell apart.
    """
    theta = torch.minimum(torch.maximum(theta, lo.clamp_min(1e-7)), hi)
    col = (w.unsqueeze(1) * Z).sum(0)
    if shrink is not None:
        col = col * (1.0 + shrink)
    for _ in range(iters):
        mu = (Z @ theta).clamp_min(1e-12)
        new = theta * ((w * y / mu).unsqueeze(1) * Z).sum(0) / col.clamp_min(1e-12)
        new = torch.where(col > 0, new, theta)
        new = torch.minimum(torch.maximum(new, lo.clamp_min(1e-7)), hi)
        done = bool(((new - theta).abs() <= 1e-4 * theta.clamp_min(1e-6)).all())
        theta = new
        if done:
            break
    return theta


def _robust_plane(y, Z, lab, incl, theta, lo, hi, k, max_ratio: float = 1.5, shrink=None):
    """Fit the plane to cells, ignoring cells and clusters that carry more than background.

    Cells: a cell whose silent-gene count is far above the fit (more than 2.5 times the expected
    count plus 3, and beyond the 0.1% Poisson tail) is a doublet or a misplaced cell, and is left
    out. Only the upper side is trimmed: real expression can only add counts.
    Clusters: a cluster whose kept cells still hold more than max_ratio times the fitted counts
    is dropped, worst first, while at least two clusters and 40% of the cells remain.
    """
    use = incl.clone()
    n0 = int(incl.sum())
    dropped = []
    while True:
        idx = torch.nonzero(use, as_tuple=True)[0]
        if idx.numel() < 200 or float(y[idx].sum()) < 500.0:
            return None
        yy, ZZ, ll = y[idx], Z[idx], lab[idx]
        w = torch.ones_like(yy)
        for _ in range(8):
            theta = _poisson_plane(yy, ZZ, w, theta, lo, hi, shrink=shrink)
            mu = (ZZ @ theta).clamp_min(1e-9)
            tail = torch.special.gammainc(yy.clamp_min(1e-6), mu)  # P(Poisson(mu) >= y) for y >= 1
            w_new = (~((yy > 2.5 * mu + 3.0) & (tail < 1e-3))).to(yy.dtype)
            if bool((w_new == w).all()):
                break
            w = w_new
        num = torch.zeros(k, dtype=yy.dtype, device=yy.device).index_add_(0, ll, w * yy)
        den = torch.zeros(k, dtype=yy.dtype, device=yy.device).index_add_(0, ll, w * mu)
        cnt = torch.bincount(ll, minlength=k)
        F = torch.where(cnt > 0, num / den.clamp_min(1e-9), torch.zeros_like(num))
        worst = int(torch.argmax(F))
        if float(F[worst]) > max_ratio and int((cnt > 0).sum()) > 2 and idx.numel() - int(cnt[worst]) >= 0.4 * n0:
            use &= lab != worst
            dropped.append(worst)
            continue
        kept = torch.zeros(y.numel(), dtype=torch.bool, device=y.device)
        kept[idx[w > 0]] = True
        return {"theta": theta, "n_cells": int(idx.numel()), "n_trimmed": int((w == 0).sum()), "kept": kept,
                "counts": float((w * yy).sum()), "ratio": {int(c): round(float(F[c]), 3) for c in torch.nonzero(cnt > 0, as_tuple=True)[0]},
                "dropped": dropped}


@torch.no_grad()
def _swap_shape(X, A, A2, D, cand_rate, on, theta, need, ridge_frac: float = 0.1, steps: int = 11, window: float = 0.15, slack: float = 0.02, s_cap: float = 0.3):
    """Which profile the depth-proportional background has, from gene contrast.

    Background of gene g in cluster c = kappa * A + kappa_d * A2 + s * D: A and A2 carry the soup profile, D the
    average cell profile (swapped molecules). Summed over a cell's silent genes, the last two cannot be told apart:
    both grow with depth. Gene by gene they can, because a gene that cells hold more of than the soup does (nuclear
    and mitochondrial RNA) carries more background than the soup predicts exactly when s > 0. Fitting that directly
    fails, though: the silent genes are chosen as those near the lowest ridge of observed over expected, the genes
    that would show a larger s lie above the ridge while s is too small, and the fit confirms where it started
    (with the truth at s = 3% it settled at 1.6% whatever the solver; on a 10x cell-line mixture at 1.2% where the
    cross-species counts give 1.6%, 2026-11-18).

    So the split is searched, not fitted. The depth-proportional background that the current fit puts on the pairs
    `on` is kept, and shared between the two profiles in every proportion w (0: all soup profile, 1: all cell
    profile). For each w the candidate pairs are compared with their expected background, and the w at which the
    most pairs lie within `window` dex of the lowest ridge wins: with the right profile the silent genes line up
    on one level, with the wrong one they smear by how much more of each gene cells hold than the soup. Expression
    lines up at no w. Among proportions within `slack` of the best the smallest is taken.

    The soup profile stays (w = 0) unless the best proportion puts clearly more pairs on the ridge than it does: by
    two standard deviations of a count and by 3%. Libraries with a few hundred candidate pairs cannot show that,
    and following their best proportion made the estimate jump (a Parse sublibrary went from 87% to 66% of its
    cross-species counts removed on a difference of 6 pairs in 104, 2026-11-18). A swapped fraction above s_cap is
    held there and the rest stays on the soup profile.

    X, A, A2, D [k, G]; cand_rate [k, G]: the cluster's rate is at most delta times the top cluster's (and the
    cluster is large enough). Returns (kappa_d, s, table) or None when there is nothing to split.
    """
    base = theta[0] * A
    dp = theta[1] * A2 + theta[2] * D
    T = float((dp * on).sum())
    SA2, SD = float((A2 * on).sum()), float((D * on).sum())
    if T <= 0 or SA2 <= 0 or SD <= 0:
        return None
    table, best = [], None
    for i in range(steps):
        w = i / (steps - 1)
        kd, sw = (1.0 - w) * T / SA2, w * T / SD
        B = base + kd * A2 + sw * D
        c = cand_rate & (B >= need)
        if int(c.sum()) < 20:
            continue
        lr = torch.log10((X[c] / B[c].clamp_min(1e-12)).clamp_min(1e-12))
        level, _ = _leftmost_peak(lr, frac=ridge_frac)
        if level is None:
            continue
        n_in = int(((lr - math.log10(level)).abs() <= window).sum())
        table.append((round(w, 2), n_in, round(level, 3)))
        if best is None or n_in > best[1]:
            best = (w, n_in, kd, sw)
    if best is None:
        return None
    w = min(float(w_) for w_, n_in, _ in table if n_in >= (1.0 - slack) * best[1])
    n0 = next((n_in for w_, n_in, _ in table if w_ == 0.0), None)
    clear = n0 is not None and best[1] - n0 >= max(2.0 * math.sqrt(max(n0, 1.0)), 0.03 * n0)
    if not clear:
        w = 0.0
    sw = min(w * T / SD, s_cap)
    kd = (T - sw * SD) / SA2
    return kd, sw, {"swap_share_of_depth_proportional": w, "swap_share_is_clear": bool(clear), "pairs_on_the_ridge_by_share": table}


def silent_budget(
    csr,
    rows: torch.Tensor,
    z: torch.Tensor,
    depth: torch.Tensor,
    unit: torch.Tensor,
    pool: torch.Tensor,
    e: torch.Tensor,
    soup: torch.Tensor,
    chibar: torch.Tensor,
    s_ref: torch.Tensor,
    n_clusters: int = 20,
    min_cells: int = 30,
    min_expected: float = 30.0,
    delta: float = 0.2,
    s_max: float = 0.1,
    kappa_min: float = 0.1,
    kappa_max: float = 20.0,
    depth_max: float = 50.0,
    ridge_frac: float = 0.1,
    rounds: int = 3,
    seed: int = 0,
    broad: bool = True,
    broad_tail: float = 0.05,
    retry: bool = True,
    seeds: int = 3,
    s_start: float = 2e-4,
    wide_level: float = -1.0,
    ridge_max: float = 5.0,
    tree: bool = True,
    tree_children: int = 8,
    tree_min_cells: int = 40,
    tree_peel: int = 0,
    tree_carriers: float = 0.25,
    splice: dict | None = None,
    swap_contrast: bool = True,
    swap_gate: float = 0.05,
    fixed_candidates: bool = False,
    cross_check: bool = True,
    cross_ratio: float = 1.5,
    factor_max: float = 1.0,
    superpose: bool = False,
    superpose_share: float = 0.2,
    superpose_match: float = 0.5,
    superpose_genes: int = 50,
):
    """Label-free background budget, fitted cell by cell on genes each cell's cluster does not express.

        background of gene g in cell n  =  (kappa + kappa_d * N_n / N_ref) * e * soup_g  +  s * N_n * chibar_g

    Three amounts, each read off the data and then held fixed, because the likelihood cannot pin
    any of them from below (the decoder can absorb background):
      kappa     ambient RNA that every partition carries, in units of the empty-partition level e
      kappa_d   ambient-profile RNA that grows with the cell's depth (N_ref is the median depth of
                the confident cells). Nuclei carry it: in the kidney truth set the background of a
                nucleus rises by 0.17 counts per count of depth while that of a whole cell is flat,
                and in a nuclei barnyard a nucleus carries about nine empty-partition levels.
      s         the swapped fraction: counts with the pool's average cell profile, proportional
                to depth (chimeric reads; dominant in clean cell-line mixtures).
    kappa_d and s both grow with depth and differ only in profile, so where the soup resembles
    the average cell the split between them is arbitrary and harmless. The upper bounds are only
    guards: e is the level of an empty partition, and a nucleus can carry many times that (the
    first bounds of 4 and 10 were both reached on a nuclei barnyard, which then lost only 59% of
    its cross-species counts).

    Version 0.3.0 first fitted a line through one summary point per cluster. On tissue that read
    the budget 1.2 to 1.8 times too high (Janssen kidney, 2026-11-16): a few percent of doublets
    or misplaced cells in a cluster add real expression of the "silent" genes, and a cluster sum
    cannot tell that from background. This version works per cell:

      1. Clusters: k-means on the latent space.
      2. Silent genes of a cluster: genes expressed there at no more than delta times their rate in
         the cluster that expresses them most, with at least min_expected background counts
         expected, and whose observed over expected background is at the lowest peak of that ratio
         over the whole library, or below it. Real expression only lifts a gene above the
         background level, so the lowest peak is the silent genes, and one level for the library
         leaves out clusters whose "silent" genes sit higher because of doublets.
      3. For every cell: y = its counts on its cluster's silent genes, and the counts expected there
         per unit of each of the three amounts. A Poisson fit over all cells, robust to cells and
         clusters that carry real expression (see _robust_plane).
      4. Steps 2 and 3 repeat with the new values, since "expected background" depends on them.
         The lowest ridge may be a small group (a tenth of the tallest peak, ridge_frac): once
         the swapped fraction is set too high, weakly expressed genes fall below 1 and outnumber
         the truly silent ones, and a rule that wants a large group then follows them upward
         (kidney nuclei: two self-consistent answers, one with a fifth too much background). The
         fit also leans slightly (3%) against the swapped term, which decides between it and the
         depth-proportional ambient term where the data cannot.

      5. The wider silent set (broad). Step 2 can only test genes with enough expected background
         (min_expected), which on a clean library is a handful: 18 genes per cluster on BD
         Rhapsody, 5 on a xenograft, against 169 on a 10x cell-line mixture (all-tier benchmark,
         2026-11-17). Those few genes fix the level, but they are too few to see which cells carry
         several times more background than the rest, and too few for the exclusion constraint to
         matter. So, with the level fixed, every rarer gene is tested the exact way: it is silent
         in a cluster when the cluster expresses it at no more than delta times the top cluster
         and its counts there are no more than background alone explains (inside the lower
         1 - broad_tail of a Poisson with the measured background as mean). Both sets together
         go to cell_multipliers and to the decoder's exclusion constraint.
         Gene contrast (swap_contrast, 0.4.0). Summed over a cell's silent genes, kappa_d and s cannot be told
         apart: both grow with depth, and the fit left s where it started (zero on every cell-line mixture
         looked at, where the cross-species counts ask for 0.3 to 1.6% of depth, 2026-11-18). The split is
         read from the genes instead (_swap_shape), from the second round on and with the cells the fit set
         aside left out of the cluster sums; the fit then has two amounts to find, the constant one and the
         depth-proportional one in that shape.
      6. No signal (retry): the clustering is repeated with half as many clusters, down to two.
         Fewer, larger clusters expect more background counts per gene, which is what a small
         or clean library lacks (one Parse sublibrary and one 1,000-cell library had no estimate
         at all in the benchmark).

    Returns (kappa per pool, s per pool, diagnostics, state per pool); an entry is None where
    fewer than two clusters show silent genes or too few cells or counts remain. The state holds
    kappa_d, N_ref, the cluster centres and each cluster's silent genes, for cell_multipliers.
    No labels are used.
    """
    n_pools, G = chibar.shape
    U = soup.shape[0]
    dev = z.device
    f64 = torch.float64
    amb = (e.unsqueeze(1) * soup).to(f64)
    soup64 = soup.to(f64)
    kappas, swaps, diag, states = [], [], {}, []
    for p in range(n_pools):
        sel = torch.nonzero(pool == p, as_tuple=True)[0]
        k = int(min(n_clusters, sel.numel() // (4 * min_cells)))
        # Small or clean libraries: fewer, larger clusters, so that a cluster still expects enough
        # background counts per gene to show its silent genes (about 15,000 ambient counts each).
        if sel.numel():
            k = int(min(k, max(2, float(e[unit[sel]].sum()) // 15000)))
        info = {"n_cells": int(sel.numel()), "n_clusters": k}
        diag[str(p)] = info
        states.append(None)
        if k < 2:
            kappas.append(None)
            swaps.append(None)
            info["reason"] = "too few confident cells"
            continue
        d = depth[sel].to(f64)
        u_sel = unit[sel]
        ks = [k]
        while retry and ks[-1] > 2:
            ks.append(max(2, ks[-1] // 2))
        # Several clusterings of the same cells (seeds): the estimate rests on which genes come out
        # silent, and on clean or small libraries that changed from run to run (the level of one
        # xenograft moved from 0.26 to 0.41 between two runs of the same code, and one Parse
        # sublibrary had an estimate in one run and none in the next). The clustering whose
        # background per cell at the median depth is the middle one is kept.
        e_bar = float(e[u_sel].to(f64).mean())
        cands = []
        # Two clusters must show silent genes, so that one impure cluster cannot set the level alone. When no
        # clustering offers that (small split-pool libraries: an estimate in one run and none in the next, and
        # without one the removal fell from 90% to 56%), a single cluster is accepted as a last resort. The
        # level is the lowest peak of observed over expected counts, so it comes from the purest cluster and
        # errs toward too little background, never too much.
        # Two rules for which pairs are tested, each from every starting clustering (cross_check). The pairs can follow
        # the fit (those that expect enough background under the current amounts) or be settled at the start (one
        # empty-partition level per cell). Each rule runs away on some library: following the fit, a larger
        # background makes more weakly expressed genes testable and they pull it up (a BD Rhapsody PBMC library:
        # 17% of depth, a quarter of the cells set aside; 3.7% from two of three clusterings in 0.3.2); settled at
        # the start, the same happened on a Parse bone marrow library (34% of depth against 2.8%). Neither did on
        # the other 24 libraries looked at, where the two rules agree within a fifth (2026-11-18). Expression can
        # only add counts to genes taken for silent, so where the two rules differ by more than half, the
        # lower answer is the one that did not run away, and it is kept.
        modes = (False, True) if cross_check else (bool(fixed_candidates),)
        for min_groups in ((2, 1) if retry else (2,)):
            for sd in range(seed, seed + max(1, int(seeds))):
                for fixed_now in modes:
                    info_s = {}
                    for k in ks:
                        lab = kmeans(z[sel], k, seed=sd)
                        X = csr.group_sums(rows[sel], lab, k)
                        n_cu = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, lab * U + u_sel, torch.ones_like(d)).view(k, U)
                        A = n_cu @ amb
                        n_ref = float(torch.median(d))
                        n2_cu = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, lab * U + u_sel, d / n_ref).view(k, U)
                        A2 = n2_cu @ amb  # ambient-profile counts per unit kappa_d
                        Ntot = torch.zeros(k, dtype=f64, device=dev).index_add_(0, lab, d)
                        chi = chibar[p].to(f64)
                        D = Ntot.unsqueeze(1) * chi.unsqueeze(0)
                        sizes = torch.bincount(lab, minlength=k)
                        big = sizes >= min_cells
                        rate = X / Ntot.clamp_min(1.0).unsqueeze(1)
                        rmax = torch.where(big.unsqueeze(1), rate, torch.zeros_like(rate)).max(0).values
                        # The swapped fraction starts small. It and the depth-proportional ambient term both grow with
                        # depth, and on a library with few silent genes the fit cannot tell them apart and moves little
                        # from where it starts. Starting at the learned value (0.6% of depth) left that much swapped
                        # background in place on clean tissue with nothing measured to support it (xenograft skin,
                        # 2026-11-17); a small start lets it grow only where the silent-gene counts ask for it.
                        s0 = float(min(max(float(s_ref[p]), 0.002), s_max)) if s_start < 0 else float(min(s_start, s_max))
                        theta = torch.tensor([1.0, 0.05, s0], dtype=f64, device=dev)
                        lo = torch.tensor([kappa_min, 0.0, 0.0], dtype=f64, device=dev)
                        hi = torch.tensor([kappa_max, depth_max, s_max], dtype=f64, device=dev)
                        shrink = torch.tensor([0.0, 0.0, 0.03], dtype=f64, device=dev)
                        fit, table = None, []
                        shape, kept_cells = None, None
                        B_start = theta[0] * A + theta[1] * A2 + theta[2] * D
                        for rnd in range(rounds + 2 if swap_contrast else rounds):
                            theta_before = theta.clone()
                            B = theta[0] * A + theta[1] * A2 + theta[2] * D
                            rho = X / B.clamp_min(1e-12)
                            B_test = B_start if fixed_now else B
                            for need in (min_expected, min_expected / 3.0):  # a second, looser look before giving up
                                cand = big.unsqueeze(1) & (rate <= delta * rmax.unsqueeze(0)) & (B_test >= need)
                                level, hist = _leftmost_peak(torch.log10(rho[cand].clamp_min(1e-12)), frac=ridge_frac)
                                if level is not None:
                                    break
                            # Swapped molecules are a few percent of a cell at most. Where the depth-proportional background is
                            # more than swap_gate of depth it is ambient RNA that the nucleus carries, and it keeps the soup
                            # profile: on kidney nuclei the search put 13 to 14% of depth on the average nucleus profile
                            # (twice as many candidate pairs on the ridge), and by the genotype truth that removed 1.45 times
                            # the true background on one library and halved the rank agreement on the other. Related cell
                            # types express each other's genes weakly and in the proportions of the average cell, which a
                            # cell-line mixture cannot show.
                            dp_share = float(theta[1]) * e_bar / n_ref + float(theta[2])
                            gate_open = swap_contrast and dp_share <= swap_gate
                            if rnd >= 1:
                                info_s["depth_proportional_background_share_of_depth"] = round(dp_share, 4)
                            if gate_open and level is not None and rnd >= 1:
                                # which profile the depth-proportional background has (_swap_shape), on cluster sums without
                                # the cells the last fit set aside: a doublet adds counts with the profile of other cells
                                Xs, As, A2s, Ds = X, A, A2, D
                                if kept_cells is not None and not bool(kept_cells.all()):
                                    kc = kept_cells
                                    Xs = csr.group_sums(rows[sel][kc], lab[kc], k)
                                    n_s = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, (lab * U + u_sel)[kc], torch.ones_like(d[kc])).view(k, U)
                                    n2_s = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, (lab * U + u_sel)[kc], (d / n_ref)[kc]).view(k, U)
                                    N_s = torch.zeros(k, dtype=f64, device=dev).index_add_(0, lab[kc], d[kc])
                                    As, A2s, Ds = n_s @ amb, n2_s @ amb, N_s.unsqueeze(1) * chi.unsqueeze(0)
                                rate_s = Xs / Ds.sum(1, keepdim=True).clamp_min(1e-12)  # D sums to the cluster's depth
                                rmax_s = torch.where(big.unsqueeze(1), rate_s, torch.zeros_like(rate_s)).max(0).values
                                cr = big.unsqueeze(1) & (rate_s <= delta * rmax_s.unsqueeze(0))
                                got = _swap_shape(Xs, As, A2s, Ds, cr, cand & (rho <= level * 10.0**0.15), theta, need, ridge_frac=ridge_frac, s_cap=s_max)
                                if got is not None:
                                    theta = torch.stack([theta[0], torch.as_tensor(min(got[0], depth_max), dtype=f64, device=dev),
                                                         torch.as_tensor(min(got[1], s_max), dtype=f64, device=dev)])
                                    shape = got[2]
                                    B = theta[0] * A + theta[1] * A2 + theta[2] * D
                                    rho = X / B.clamp_min(1e-12)
                                    cand = big.unsqueeze(1) & (rate <= delta * rmax.unsqueeze(0)) & ((B_start if fixed_now else B) >= need)
                                    lvl2, hist2 = _leftmost_peak(torch.log10(rho[cand].clamp_min(1e-12)), frac=ridge_frac)
                                    if lvl2 is not None:
                                        level, hist = lvl2, hist2
                            info_s["min_expected_used"] = float(need)
                            info_s["ridge"] = None if level is None else round(level, 3)
                            info_s["ridge_histogram"] = [int(v) for v in hist]
                            mask = torch.zeros(k, G, dtype=torch.bool, device=dev)
                            if level is not None:
                                mask = cand & (rho <= level * 10.0**0.15)
                            table = [{"cluster": c, "cells": int(sizes[c]), "candidates": int(cand[c].sum()), "genes": int(mask[c].sum())} for c in range(k)]
                            if int(mask.any(1).sum()) < min_groups:
                                fit = None
                                break
                            mk = mask.to(f64)
                            y = csr.masked_row_sums(rows[sel], lab, mask)
                            a = e[u_sel].to(f64) * (soup64 @ mk.T)[u_sel, lab]
                            dd = d * (mk @ chi)[lab]
                            if gate_open and rnd >= 1 and shape is not None:
                                # two amounts: the constant one, and the depth-proportional one in the shape just read
                                col = theta[1] * a * d / n_ref + theta[2] * dd
                                Z = torch.stack([a, col], dim=1)
                                fit = _robust_plane(y, Z, lab, mask.any(1)[lab], torch.stack([theta[0], torch.ones((), dtype=f64, device=dev)]),
                                                    lo[:2].clone(), torch.tensor([kappa_max, 1e3], dtype=f64, device=dev), k)
                                if fit is not None:
                                    m_dp = fit["theta"][1]
                                    s_new = (theta[2] * m_dp).clamp(max=s_max)
                                    # what the cap takes from the swapped part goes to the soup-profile part, count for count
                                    spill = (theta[2] * m_dp - s_new) * dd.sum() / (a * d / n_ref).sum().clamp_min(1e-12)
                                    fit["theta"] = torch.stack([fit["theta"][0], (theta[1] * m_dp + spill).clamp(max=depth_max), s_new])
                            else:
                                Z = torch.stack([a, a * d / n_ref, dd], dim=1)
                                fit = _robust_plane(y, Z, lab, mask.any(1)[lab], theta, lo, hi, k, shrink=shrink)
                            if fit is None:
                                break
                            theta = fit["theta"]
                            kept_cells = fit["kept"] | ~mask.any(1)[lab]
                            if rnd + 1 >= rounds and (not gate_open or bool(((theta - theta_before).abs() <= 0.02 * theta_before.clamp_min(1e-4)).all())):
                                break
                            if not gate_open:
                                shape = None
                        info_s["fit_rounds"] = rnd + 1
                        if shape is not None:
                            info_s.update(shape)
                        # The level is read against the fit of the round before, so at the end it sits near 1 when the
                        # two agree (1.1 to 3.6 on every development library). Where it does not, the "silent" group
                        # of this clustering is a group of weakly expressed genes: on one Parse tissue library such
                        # clusterings ended at 8.9 and 11, and when two of the three seeds did, 11 to 17% of the
                        # counts were removed instead of 2%. Such a clustering does not count; fewer clusters are tried.
                        if fit is not None and level is not None and level > ridge_max:
                            info_s["not_converged_ridge"] = round(level, 3)
                            fit = None
                        if fit is not None:
                            break
                    if fit is not None:
                        cands.append({"lab": lab, "k": k, "theta": theta, "fit": fit, "table": table, "level": level, "need": need, "B": B,
                                      "mask": mask, "big": big, "n_ref": n_ref, "chi": chi, "info": info_s, "seed": sd,
                                      "value": e_bar * float(theta[0] + theta[1]) + float(theta[2]) * n_ref, "groups": min_groups, "fixed": fixed_now})
            if cands:
                break
        if cands:
            cands.sort(key=lambda c: c["value"])
            mids = {}
            for m_ in modes:
                grp = [c for c in cands if c["fixed"] == m_]
                if grp:
                    mids[m_] = grp[(len(grp) - 1) // 2]
            rule = modes[0] if modes[0] in mids else next(iter(mids))
            if len(mids) == 2:
                lo_rule = min(mids, key=lambda m_: mids[m_]["value"])
                hi_rule = not lo_rule
                info["rules_disagree"] = bool(mids[hi_rule]["value"] > cross_ratio * mids[lo_rule]["value"])
                if info["rules_disagree"]:
                    rule = lo_rule
                info["background_by_seed_other_rule"] = [round(c["value"], 2) for c in cands if c["fixed"] != rule]
            info["tested_pairs_rule"] = "settled at the start" if rule else "follow the fit"
            best = mids[rule]
            cands = [c for c in cands if c["fixed"] == rule]
            lab, k, theta, fit, table, level, need, B = (best[x] for x in ("lab", "k", "theta", "fit", "table", "level", "need", "B"))
            mask, big, n_ref, chi = (best[x] for x in ("mask", "big", "n_ref", "chi"))
            info.update(best["info"])
            info["seeds_with_signal"] = len(cands)
            if best["groups"] < 2:
                info["single_cluster_estimate"] = True
            info["background_at_median_depth_by_seed"] = [round(c["value"], 2) for c in cands]
        else:
            info.update(info_s)
            info["seeds_with_signal"] = 0
        info["n_clusters"] = k
        if len(ks) > 1:
            info["clusters_tried"] = ks[: ks.index(k) + 1]
        info["clusters"] = table
        if fit is None:
            kappas.append(None)
            swaps.append(None)
            info["reason"] = "no silent-gene signal"
            continue
        for row in table:
            row["ratio"] = fit["ratio"].get(row["cluster"])
            row["used"] = row["genes"] > 0 and row["cluster"] not in fit["dropped"]
        kap, kap_d, s_fit = (float(x) for x in theta)
        info.update({"kappa": kap, "kappa_depth": kap_d, "s": s_fit, "depth_ref": n_ref, "cells_fitted": fit["n_cells"],
                     "cells_trimmed": fit["n_trimmed"], "silent_counts": fit["counts"], "clusters_dropped": fit["dropped"]})
        kappas.append(kap)
        swaps.append(s_fit)
        zsel = z[sel].to(f64)
        cnt = torch.bincount(lab, minlength=k).to(f64).unsqueeze(1)
        centres = torch.zeros(k, zsel.shape[1], dtype=f64, device=dev).index_add_(0, lab, zsel) / cnt.clamp_min(1.0)
        keep_c = torch.tensor([c not in fit["dropped"] for c in range(k)], device=dev)
        states[-1] = {"centres": centres, "mask": mask & keep_c.unsqueeze(1), "kappa": kap, "kappa_depth": kap_d, "depth_ref": n_ref, "s": s_fit}
        if broad and tree and tree_peel > 0:
            # Multi-resolution silent sets, first version (multires.tree_silent): every cluster is split into sub-clusters; a gene one or two
            # sub-clusters express is silent for the rest, and a sub-cluster that holds far more of its parent's
            # counts than its share of the background overrules the parent. Isolated carriers do not count.
            from .multires import subclusters, tree_silent

            fine, parent, fcentres = subclusters(z[sel], lab, k, min_cells=tree_min_cells, max_children=tree_children, seed=seed)
            sp_arg = None
            if splice is not None:
                # Spliced against unspliced molecules (splice.py): per-gene unspliced shares of background and of
                # own expression, measured on the empty partitions, the tested silent genes and each gene's top cluster.
                from .splice import splice_profiles, sums_of

                bg = splice["bg_rows"][splice["bg_pool"] == p]
                zero = torch.zeros(bg.numel(), dtype=torch.long, device=dev)
                Xe = csr.group_sums(bg, zero, 1)[0] if bg.numel() else torch.zeros(G, dtype=f64, device=dev)
                Ue = sums_of(csr, splice["u"], bg, zero, 1)[0] if bg.numel() else torch.zeros(G, dtype=f64, device=dev)
                Xk = csr.group_sums(rows[sel], lab, k)
                Uk = torch.minimum(sums_of(csr, splice["u"], rows[sel], lab, k), Xk)
                prof = splice_profiles(Xe, Ue, Xk, Uk, B, mask & keep_c.unsqueeze(1), big, level, shift=splice.get("shift", 0.0))
                info["splice"] = prof["diag"]
                states[-1]["splice"] = {key: prof[key] for key in ("pi_bg", "pi_nat", "informative")}
                if splice.get("silent", True) and int(prof["informative"].sum()) >= 50:
                    sp_arg = {"u": splice["u"], **states[-1]["splice"]}
                del Xk, Uk
            silent, strip, rate_f, _, tdiag = tree_silent(csr, rows[sel], fine, parent, k, d, u_sel, e, soup64, chi, theta, n_ref, level, need,
                                                          keep_c, big, delta=delta, tail=broad_tail, max_peel=tree_peel, splice=sp_arg)
            states[-1].update({"centres": fcentres, "mask": (mask & keep_c.unsqueeze(1))[parent] & silent, "mask_broad": silent,
                               "rate": rate_f, "parent": parent})
            if sp_arg is not None:
                states[-1]["mask_strip"] = strip  # also the genes whose molecules are spliced like background
            info.update(tdiag)
            info["silent_genes_tested_median"] = float(torch.median(mask.sum(1).to(f64)))
            info["silent_genes_broad_median"] = float(torch.median(silent.sum(1).to(f64)))
            info["silent_counts_broad"] = tdiag["silent_counts"]
        elif broad:
            # B and level are those of the last round, the ones the tested set above was read against.
            # Cells whose silent-gene counts are far above the fit (doublets, misplaced cells) are left
            # out of the cluster sums first: a handful of them express a gene for real and would keep it
            # out of the set for the whole cluster (in a cell-line mixture, ten human + mouse doublets
            # among 500 human cells put every mouse gene far above background). Two passes, since the
            # wider set shows such cells better than the few tested genes do.
            rows_sel = rows[sel]
            wide = mask & keep_c.unsqueeze(1)
            scale = 1.0
            for _ in range(2):
                mkw = wide.to(f64)
                yw = csr.masked_row_sums(rows_sel, lab, wide)
                aw = e[u_sel].to(f64) * (soup64 @ mkw.T)[u_sel, lab]
                dw = d * (mkw @ chi)[lab]
                muw = (scale * (theta[0] * aw + theta[1] * aw * d / n_ref + theta[2] * dw)).clamp_min(1e-9)
                odd = (yw > 2.5 * muw + 3.0) & (torch.special.gammainc(yw.clamp_min(1e-6), muw) < 1e-3)
                ok = ~odd
                Xc = csr.group_sums(rows_sel[ok], lab[ok], k)
                n_c = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, (lab * U + u_sel)[ok], torch.ones_like(d[ok])).view(k, U)
                n2_c = torch.zeros(k * U, dtype=f64, device=dev).index_add_(0, (lab * U + u_sel)[ok], (d / n_ref)[ok]).view(k, U)
                Nc = torch.zeros(k, dtype=f64, device=dev).index_add_(0, lab[ok], d[ok])
                Bc = theta[0] * (n_c @ amb) + theta[1] * (n2_c @ amb) + theta[2] * Nc.unsqueeze(1) * chi.unsqueeze(0)
                rate_c = Xc / Nc.clamp_min(1.0).unsqueeze(1)
                rmax_c = torch.where(big.unsqueeze(1), rate_c, torch.zeros_like(rate_c)).max(0).values
                # What "background alone explains" is the fitted background itself (wide_level = 1). Up to 0.3.1
                # the test used the ridge level instead, the lowest peak of observed over fitted counts in the
                # cluster sums. That peak sits 1.1 to 1.8 times above the fit because the sums include cells
                # the fit leaves out; here those cells are already out of the sums, so using it again let a gene
                # with up to 1.8 times the fitted background be called silent and stripped (the over-cleaning
                # of clean tissue, 2026-11-17). A negative wide_level restores the ridge level.
                lv = level if wide_level < 0 else wide_level
                lam = (lv * Bc).clamp_min(1e-12)
                fits_background = (Xc < 0.5) | (torch.special.gammainc(Xc.clamp_min(1e-6), lam) >= broad_tail)  # P(Poisson(lam) >= X)
                tested = (Bc >= need) & (Xc <= lv * 10.0**0.15 * Bc)
                rare = (Bc < need) & fits_background
                wide = (mask | (big.unsqueeze(1) & (rate_c <= delta * rmax_c.unsqueeze(0)) & (tested | rare))) & keep_c.unsqueeze(1)
                inc = ok & (aw > 0)
                if int(inc.sum()) >= 50:  # the wider set's expected counts, rescaled to what its cells show
                    scale = float(min(max(float(yw[inc].sum() / (muw[inc].sum() / scale).clamp_min(1e-9)), 0.25), 4.0))
            # The gene's own background level, read across clusters (multires.shared_level): a gene that sits at the same
            # multiple of the fit in every cluster that does not express it is silent there, at that multiple.
            phi = torch.ones(G, dtype=f64, device=dev)
            if factor_max > 1.0:
                from .multires import shared_level

                cand_c = (big & keep_c).unsqueeze(1) & (rate_c <= delta * rmax_c.unsqueeze(0))
                member, phi = shared_level(Xc, lv * Bc, cand_c, need, broad_tail, factor_max)
                up, dn = phi > 1.25, phi < 0.8
                info.update({"genes_with_shared_level_above_1.25": int(up.sum()), "genes_with_shared_level_below_0.8": int(dn.sum()),
                             "pairs_silent_by_shared_level": int((member & ~wide).sum()),
                             "counts_silent_by_shared_level": float((Xc * (member & ~wide).to(f64)).sum()),
                             "fitted_background_on_genes_with_a_level": float((lv * Bc * (phi != 1.0).unsqueeze(0)).sum()),
                             "the_same_at_their_levels": float((lv * Bc * phi.unsqueeze(0) * (phi != 1.0).unsqueeze(0)).sum())})
                wide = wide | member
            # A cluster that is another cluster plus background-like RNA is not a cell type. Cells that carry several
            # times the usual background cluster apart (on a BD Rhapsody cell-line mixture, 80 mouse cells with 6% human
            # RNA made a cluster in which every human gene counted as expressed and was kept, 2026-11-18). The rule
            # that 0.3.2 applies to single cells is applied to clusters: where the silent genes of a larger cluster
            # hold at least three times this cluster's fitted background and at most a fifth of its counts, on many
            # genes and in the proportions of the background, the cluster takes the larger cluster's silent genes.
            # Its cells are then judged one by one as before (cell_multipliers, background_match). A cell type of its
            # own has those counts on genes of its own: to reach three times the background its program must hold
            # two thirds of them, and then they do not follow the background's proportions.
            merged_into = {}
            if superpose:
                sizes = torch.bincount(lab, minlength=k)
                Bphi = lv * Bc * phi.unsqueeze(0)
                usable = big & keep_c
                rec, near = {}, []
                for c in range(k):
                    if not bool(big[c]):
                        continue
                    for a_ in range(k):
                        if a_ == c or not bool(usable[a_]) or int(sizes[a_]) <= int(sizes[c]) or not bool((wide[a_] & ~wide[c]).any()):
                            continue
                        S = wide[a_]
                        yv, qv = Xc[c] * S, Bphi[c] * S
                        Yt, Qt = float(yv.sum()), float(qv.sum())
                        n_g = int((yv > 0).sum())
                        if n_g < superpose_genes or Qt <= 0 or Yt < 3.0 * Qt:
                            continue
                        share = Yt / max(float(Nc[c]), 1.0)
                        ov = float(torch.minimum(yv / Yt, qv / Qt).sum())
                        row = {"cluster": c, "like": a_, "cells": int(sizes[c]), "share_of_counts": round(share, 4), "over_background": round(Yt / Qt, 2),
                               "match": round(ov, 3), "genes": n_g}
                        if share <= superpose_share and ov >= superpose_match:
                            if c not in rec or ov > rec[c]["match"]:
                                rec[c] = row
                        else:
                            near.append(row)
                for c, row in rec.items():
                    root = row["like"]
                    while root in rec:  # a larger cluster each time, so this ends
                        root = rec[root]["like"]
                    merged_into[c] = root
                for c, root in merged_into.items():
                    wide[c], mask[c] = wide[root], mask[root]
                    keep_c[c] = True
                info["clusters_given_a_larger_clusters_silent_genes"] = list(rec.values())
                best_near = {}
                for r_ in near:  # per cluster kept apart, the larger cluster it resembles most
                    if r_["cluster"] not in rec and (r_["cluster"] not in best_near or r_["match"] > best_near[r_["cluster"]]["match"]):
                        best_near[r_["cluster"]] = r_
                info["clusters_far_above_background_kept_apart"] = sorted(best_near.values(), key=lambda r: -r["match"])[:10]
            states[-1]["mask"] = mask & keep_c.unsqueeze(1)
            states[-1]["phi"] = phi
            states[-1]["mask_broad"] = wide
            states[-1]["rate"] = rate_c  # counts per unit depth, cells far above the fit left out (cell_multipliers: cluster match)
            strip = None
            if splice is not None:
                # Spliced against unspliced molecules (splice.py). A gene is also silent in a cluster where its
                # molecules are spliced like background (and no more than four times the fitted background), and
                # never where they are unspliced like the gene's own expression.
                from .splice import splice_profiles, splice_support, splice_veto, sums_of

                bg = splice["bg_rows"][splice["bg_pool"] == p]
                zero = torch.zeros(bg.numel(), dtype=torch.long, device=dev)
                Xe = csr.group_sums(bg, zero, 1)[0] if bg.numel() else torch.zeros(G, dtype=f64, device=dev)
                Ue = sums_of(csr, splice["u"], bg, zero, 1)[0] if bg.numel() else torch.zeros(G, dtype=f64, device=dev)
                Uc = torch.minimum(sums_of(csr, splice["u"], rows_sel[ok], lab[ok], k), Xc)
                prof = splice_profiles(Xe, Ue, Xc, Uc, Bc, mask & keep_c.unsqueeze(1), big, level, shift=splice.get("shift", 0.0))
                info["splice"] = prof["diag"]
                states[-1]["splice"] = {key: prof[key] for key in ("pi_bg", "pi_nat", "informative")}
                if splice.get("silent", True) and int(prof["informative"].sum()) >= 50:
                    pb, pn, inf_ = prof["pi_bg"].unsqueeze(0), prof["pi_nat"].unsqueeze(0), prof["informative"].unsqueeze(0)
                    veto = splice_veto(Xc, Uc, pb, pn, inf_)
                    sup = splice_support(Xc, Uc, pb, pn, inf_) & (Xc <= 4.0 * lv * Bc) & big.unsqueeze(1) & keep_c.unsqueeze(1) & ~veto
                    info["splice"].update({"cluster_gene_pairs_silent_by_splicing_only": int((sup & ~wide).sum()),
                                           "cluster_gene_pairs_vetoed_by_splicing": int((wide & veto).sum()),
                                           "counts_silent_by_splicing_only": float((Xc * (sup & ~wide).to(f64)).sum())})
                    wide = wide & ~veto
                    strip = wide | sup
                    states[-1]["mask_broad"], states[-1]["mask_strip"] = wide, strip
            if tree:
                # Multi-resolution silent sets (multires.py): every cluster is split into sub-clusters, and a
                # sub-cluster overrules its cluster for a gene that a quarter or more of its cells carry. Doublets
                # are a few percent of a sub-cluster and overrule nothing; a sub-type or a rare cell type does.
                from .multires import subcluster_carriers, subclusters

                fine, parent, fcentres = subclusters(z[sel], lab, k, min_cells=tree_min_cells, max_children=tree_children, seed=seed)
                Fn = int(parent.numel())
                Xf, Npos, Uf = subcluster_carriers(csr, rows_sel, fine, Fn, d, u_sel, e, soup64 * phi.unsqueeze(0), chi * phi, theta, n_ref, lv,
                                                   u=splice["u"] if strip is not None else None)
                n_f = torch.bincount(fine, minlength=Fn).to(f64)
                N_f = torch.zeros(Fn, dtype=f64, device=dev).index_add_(0, fine, d)
                over = (Npos >= 3.0) & (Npos >= tree_carriers * n_f.unsqueeze(1))
                if superpose:
                    # the same question for a sub-cluster: far above its background on its cluster's silent genes, at
                    # most a fifth of its counts, in the proportions of the background, is a sub-cluster of dirty
                    # cells and overrules nothing
                    Sp = wide[parent]
                    yv = Xf * Sp
                    qp = lv * Bc * phi.unsqueeze(0) * wide  # the cluster's background on its silent genes
                    Np = torch.zeros(k, dtype=f64, device=dev).index_add_(0, lab, d)
                    Qt = qp.sum(1)[parent] * N_f / Np[parent].clamp_min(1.0)  # the sub-cluster's part, by depth
                    Yt = yv.sum(1)
                    ov = torch.minimum(yv / Yt.clamp_min(1e-12).unsqueeze(1), (qp / qp.sum(1, keepdim=True).clamp_min(1e-12))[parent]).sum(1)
                    like_bg = ((yv > 0).sum(1) >= superpose_genes) & (Yt >= 3.0 * Qt) & (Yt <= superpose_share * N_f.clamp_min(1.0)) & (ov >= superpose_match)
                    for c in merged_into:  # sub-clusters of a cluster already judged as a whole
                        like_bg |= parent == c
                    info["subclusters_background_like_not_overruling"] = int(like_bg.sum())
                    over = over & ~like_bg.unsqueeze(1)
                    del Sp, yv, qp
                if strip is not None:  # a sub-cluster whose own molecules of the gene are unspliced like expression
                    from .splice import splice_veto

                    over |= splice_veto(Xf, Uf, prof["pi_bg"].unsqueeze(0), prof["pi_nat"].unsqueeze(0), prof["informative"].unsqueeze(0))
                wide_f = wide[parent] & ~over
                info.update({"fine_clusters": Fn, "fine_cells_median": float(torch.median(n_f)),
                             "subcluster_gene_pairs_overruling_parent": int((wide[parent] & over).sum()),
                             "counts_kept_by_overruling": float((Xf * (wide[parent] & over).to(f64)).sum())})
                states[-1].update({"centres": fcentres, "mask": (mask & keep_c.unsqueeze(1))[parent] & wide_f, "mask_broad": wide_f,
                                   "rate": Xf / N_f.clamp_min(1.0).unsqueeze(1), "parent": parent})
                if strip is not None:
                    states[-1]["mask_strip"] = strip[parent] & ~over
            if os.environ.get("GLYMPH_DEBUG_DUMP"):  # development only: the silent test, pair by pair (code/benchmark/diag_silent_pairs.py)
                dbg = {"Xc": Xc, "Bc": Bc, "rate": rate_c, "rmax": rmax_c, "wide": wide, "tested_mask": mask, "keep_c": keep_c, "big": big,
                       "lv": float(lv), "level": float(level), "need": float(need), "theta": theta, "lab": lab, "sel": sel, "rows": rows_sel,
                       "odd": odd, "depth": d, "tail": broad_tail, "delta": delta, "A": n_c @ amb, "A2": n2_c @ amb, "Nc": Nc, "chi": chi}
                dbg["phi"], dbg["merged_into"] = phi, merged_into
                if tree:
                    dbg.update({"fine": fine, "parent": parent, "over": over, "wide_f": wide_f, "Xf": Xf, "Npos": Npos, "n_f": n_f})
                torch.save({k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in dbg.items()}, f"{os.environ['GLYMPH_DEBUG_DUMP']}.pool{p}.pt")
            info["silent_genes_tested_median"] = float(torch.median(mask.sum(1).to(f64)))
            info["silent_genes_broad_median"] = float(torch.median(wide.sum(1).to(f64)))
            info["silent_counts_broad"] = float((Xc * wide.to(f64)).sum())
            info["cells_left_out_of_cluster_sums"] = int(odd.sum())
    return kappas, swaps, diag, states


def _overlap(y, lam):
    """Overlap of counts y with expected counts lam (same row totals), over its expectation for Poisson draws.

    sum_g min(y_g, lam_g) against sum_g lam_g (1 - P(Poisson(lam_g) = floor(lam_g))), the exact mean of
    min(Poisson(lam), lam) gene by gene. With few counts over thousands of genes the raw overlap is far
    below one even for a perfect match; the ratio is about 1 for a match and falls toward 0 otherwise.
    """
    fl = torch.floor(lam)
    pmf = torch.exp(fl * torch.log(lam.clamp_min(1e-30)) - lam - torch.lgamma(fl + 1.0))
    want = (lam * (1.0 - pmf)).sum(1)
    got = torch.minimum(y, lam).sum(1)
    return torch.where(want > 0, got / want.clamp_min(1e-12), torch.full_like(got, float("nan")))


@torch.no_grad()
def background_match(csr, rows, lab, mask, amb_scale, soup, unit, swp_scale, chibar_p, rate=None, chunk: int = 512):
    """Two profile tests for cells with far more counts on their cluster's silent genes than background explains.

    Background match: how closely the cell's counts on the silent genes follow the fitted background
    profile there (ambient plus swapped, normalised over the silent genes). About 1 when those counts
    are background; low when they sit on genes background does not favour, which is a second cell's
    expression (a doublet) or a cell type too rare to form its own cluster.

    Cluster match (needs rate, the clusters' counts per unit depth): the share of all the cell's counts
    that its cluster's own expression explains, that is the share of its counts on the other genes
    times how closely they follow the cluster's profile there. A heavily contaminated member of the
    cluster still looks like the cluster on the genes the cluster expresses; a cell that was only
    placed there (the human immune cells of a mouse xenograft, in a mouse cluster) does not, and its
    counts on the "silent" genes are its own even when they resemble the background profile, which
    they do when the background on those genes came from cells like it.

    amb_scale and swp_scale are per row: multiplier times ambient counts per partition, and swapped
    fraction times depth. Returns (background match, cluster match), NaN where there is nothing to compare.
    """
    f64 = torch.float64
    out = torch.full((rows.numel(),), float("nan"), dtype=f64, device=rows.device)
    member = torch.full((rows.numel(),), float("nan"), dtype=f64, device=rows.device)
    chi = chibar_p.to(f64).unsqueeze(0)
    for i in range(0, rows.numel(), chunk):
        sl = slice(i, i + chunk)
        x, _ = csr.dense(rows[sl])
        x = x.to(f64)
        m = mask[lab[sl]].to(f64)
        y = x * m
        b = (amb_scale[sl].to(f64).unsqueeze(1) * soup[unit[sl]].to(f64) + swp_scale[sl].to(f64).unsqueeze(1) * chi) * m
        Y, Bt = y.sum(1, keepdim=True), b.sum(1, keepdim=True)
        o = _overlap(y, Y * b / Bt.clamp_min(1e-30))
        out[sl] = torch.where((Y.squeeze(1) > 0) & (Bt.squeeze(1) > 0), o, torch.full_like(o, float("nan")))
        if rate is not None:
            own = x - y
            pr = rate[lab[sl]].to(f64) * (1.0 - m)
            n_own, Pt = own.sum(1, keepdim=True), pr.sum(1, keepdim=True)
            o2 = _overlap(own, n_own * pr / Pt.clamp_min(1e-30))
            share = n_own.squeeze(1) / x.sum(1).clamp_min(1.0)
            member[sl] = torch.where((n_own.squeeze(1) > 0) & (Pt.squeeze(1) > 0), share * o2, torch.zeros_like(o2))
            del own, pr
        del x, m, y, b
    return out, member


@torch.no_grad()
def cell_multipliers(csr, rows, z, depth, unit, e, soup, chibar_p, state, called=None, doublet_fold: float = 5.0, doublet_share: float = 0.05, cluster_test: bool = False):
    """Ambient multiplier of each barcode: the fitted trend in depth, then the cell's own evidence.

    Every barcode starts from kappa + kappa_d * N / N_ref. The background per cell is not one
    number, though: in the kidney truth set the mean is 1.4 to 1.6 times the median. A cell's
    counts on genes its cluster does not express measure its own ambient load directly, the way
    reads of another genotype do in a strain mixture. With eps ~ Gamma(alpha, alpha / k_n) across
    cells (k_n the trend value) and y - s d ~ Poisson(eps * a), the posterior mean is
    k_n (alpha + y - s d) / (alpha + k_n a). alpha comes from how much more the counts scatter
    than Poisson. Barcodes not called cells, cells with no silent-gene evidence, and cells whose
    counts are far above the fit keep the trend value; a cell's own multiplier stays within a
    fifth to five times the trend.

    Cells with a second expression program. A cell with more than doublet_fold times the counts
    background explains on its cluster's silent genes, and more than doublet_share of all its counts
    there, is returned in the excess flag. Both conditions are needed: a nucleus in a dirty library
    has many counts on silent genes but no more than background explains, and a cell in a clean
    library can carry several times the usual background and still be a singlet. Such a cell is
    either heavily contaminated or it expresses those genes itself: a doublet, or a cell type too
    rare to form its own cluster (the human immune cells of a mouse xenograft sat in mouse clusters
    and lost 92 to 99% of their counts in 0.3.1, where every flagged cell was still held to its
    cluster's silent genes). Two things tell them apart. The gene profile: background_match is near
    1 when the cell's silent-gene counts follow the fitted background profile and low when they do
    not. And the amount: among flagged cells that passed the profile test, those with up to 15 to 20%
    of their counts on silent genes carried at least that much background by the species and genotype
    truths, and those with more lost two to twenty times their true background (development
    libraries, 2026-11-18: xenograft 27 to 36% removed against 0.6 to 1.3%; kidney 23 to 43% against
    8 to 15%). api.calibrate_background leaves a cell alone when its match is below
    Config.doublet_match or its silent share is above Config.doublet_share_max: no exclusion
    constraint, trend multiplier, glymph_doublet_like in the output. A third test, whether the cell
    looks like its cluster on the genes the cluster expresses (Config.cluster_match), changed little
    and is off. Sparing every flagged cell was tested on
    2026-11-17 and left the most contaminated cells of BD Rhapsody and Parse mixtures uncleaned
    (removal 91% to 88% and 82-90% to 67%).

    Returns (multiplier per row, diagnostics, excess flag per row, [3, rows] background match, cluster match (with cluster_test) and
    share of the cell's counts on the silent genes; NaN where not flagged).
    """
    f64 = torch.float64
    kap, kap_d, n_ref, s = float(state["kappa"]), float(state["kappa_depth"]), float(state["depth_ref"]), float(state["s"])
    dep = depth.to(f64)
    k_n = kap + kap_d * dep / n_ref
    lab = torch.cdist(z.to(f64), state["centres"]).argmin(1)
    wide = state.get("mask_broad") is not None
    mask = state["mask_broad"] if wide else state["mask"]
    y = csr.masked_row_sums(rows, lab, mask)
    mk = mask.to(f64)
    a = e[unit].to(f64) * (soup.to(f64) @ mk.T)[unit, lab]
    d = dep * (mk @ chibar_p.to(f64))[lab]
    informative = a > 0
    if called is not None:
        informative &= called
    scale = 1.0
    for _ in range(3 if wide else 1):
        mu = (k_n * a + s * d).clamp_min(1e-9)
        outlier = (y > 2.5 * mu + 3.0) & (torch.special.gammainc(y.clamp_min(1e-6), mu) < 1e-3)
        w = informative & ~outlier
        if not wide or int(w.sum()) < 50:
            break
        # The level was fixed on the tested genes. The wider set is chosen by its low counts, so
        # its expected counts are rescaled to what its cells show on average; each cell is then
        # compared with that, and only how cells differ from one another comes from this set.
        c = float(y[w].sum() / mu[w].sum().clamp_min(1e-9))
        c = min(max(c, 0.25), 4.0)
        a, d, scale = a * c, d * c, scale * c
    mu = (k_n * a + s * d).clamp_min(1e-9)
    outlier = (y > 2.5 * mu + 3.0) & (torch.special.gammainc(y.clamp_min(1e-6), mu) < 1e-3)
    w = informative & ~outlier
    diag = {"kappa": kap, "kappa_depth": kap_d, "depth_ref": n_ref, "n_informative": int(w.sum()), "n_outlier": int((outlier & informative).sum()),
            "n_barcodes": int(rows.numel()), "silent_set": "broad" if wide else "tested", "expected_rescaled_by": round(scale, 4)}
    doublet_like = informative & (y > doublet_fold * mu) & (y > doublet_share * dep) if wide else torch.zeros_like(informative)
    diag["n_silent_excess"] = int(doublet_like.sum())
    match = torch.full((3, rows.numel()), float("nan"), dtype=f64, device=z.device)  # background match, cluster match, silent share
    fl = torch.nonzero(doublet_like, as_tuple=True)[0]
    if fl.numel():
        bm, cm = background_match(csr, rows[fl], lab[fl], mask, (k_n * e[unit].to(f64))[fl], soup, unit[fl], (s * dep)[fl], chibar_p,
                                  rate=state.get("rate") if cluster_test else None)
        match[0, fl], match[1, fl], match[2, fl] = bm, cm, (y / dep.clamp_min(1.0))[fl]
        for name, v in (("background_match_q10_q50_q90", bm), ("cluster_match_q10_q50_q90", cm)):
            v = v[~torch.isnan(v)]
            if v.numel():
                diag[name] = [round(float(x), 3) for x in torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], dtype=f64, device=z.device))]
    if int(w.sum()) < 50:
        return k_n.to(torch.float32), diag, doublet_like, match
    excess = float((((y - mu) ** 2 - mu)[w]).sum())
    alpha = float(((k_n * a)[w]).pow(2).sum()) / excess if excess > 0 else 50.0
    alpha = min(max(alpha, 0.5), 50.0)
    m = k_n * (alpha + (y - s * d).clamp_min(0.0)) / (alpha + k_n * a)
    # Cells far above the fit keep the trend value. Giving them their own value (up to five times
    # the trend) was tried and removed 1.07 to 1.09 times the true background on the kidney set.
    m = torch.minimum(torch.maximum(torch.where(w, m, k_n), k_n / 5.0), k_n * 5.0)
    q = torch.quantile((m / k_n)[w], torch.tensor([0.1, 0.5, 0.9], dtype=f64, device=z.device))
    diag.update({"alpha": alpha, "expected_silent_counts_median": float(torch.median(mu[w])),
                 "multiplier_over_trend_q10_q50_q90": [round(float(x), 3) for x in q]})
    return m.to(torch.float32), diag, doublet_like, match
