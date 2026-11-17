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
        return {"theta": theta, "n_cells": int(idx.numel()), "n_trimmed": int((w == 0).sum()),
                "counts": float((w * yy).sum()), "ratio": {int(c): round(float(F[c]), 3) for c in torch.nonzero(cnt > 0, as_tuple=True)[0]},
                "dropped": dropped}


@torch.no_grad()
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
        for sd in range(seed, seed + max(1, int(seeds))):
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
                theta = torch.tensor([1.0, 0.05, float(min(max(float(s_ref[p]), 0.002), s_max))], dtype=f64, device=dev)
                lo = torch.tensor([kappa_min, 0.0, 0.0], dtype=f64, device=dev)
                hi = torch.tensor([kappa_max, depth_max, s_max], dtype=f64, device=dev)
                shrink = torch.tensor([0.0, 0.0, 0.03], dtype=f64, device=dev)
                fit, table = None, []
                for _ in range(rounds):
                    B = theta[0] * A + theta[1] * A2 + theta[2] * D
                    rho = X / B.clamp_min(1e-12)
                    for need in (min_expected, min_expected / 3.0):  # a second, looser look before giving up
                        cand = big.unsqueeze(1) & (rate <= delta * rmax.unsqueeze(0)) & (B >= need)
                        level, hist = _leftmost_peak(torch.log10(rho[cand].clamp_min(1e-12)), frac=ridge_frac)
                        if level is not None:
                            break
                    info_s["min_expected_used"] = float(need)
                    info_s["ridge"] = None if level is None else round(level, 3)
                    info_s["ridge_histogram"] = [int(v) for v in hist]
                    mask = torch.zeros(k, G, dtype=torch.bool, device=dev)
                    if level is not None:
                        mask = cand & (rho <= level * 10.0**0.15)
                    table = [{"cluster": c, "cells": int(sizes[c]), "candidates": int(cand[c].sum()), "genes": int(mask[c].sum())} for c in range(k)]
                    if int(mask.any(1).sum()) < 2:
                        fit = None
                        break
                    mk = mask.to(f64)
                    y = csr.masked_row_sums(rows[sel], lab, mask)
                    a = e[u_sel].to(f64) * (soup64 @ mk.T)[u_sel, lab]
                    dd = d * (mk @ chi)[lab]
                    Z = torch.stack([a, a * d / n_ref, dd], dim=1)
                    fit = _robust_plane(y, Z, lab, mask.any(1)[lab], theta, lo, hi, k, shrink=shrink)
                    if fit is None:
                        break
                    theta = fit["theta"]
                if fit is not None:
                    break
            if fit is not None:
                cands.append({"lab": lab, "k": k, "theta": theta, "fit": fit, "table": table, "level": level, "need": need, "B": B,
                              "mask": mask, "big": big, "n_ref": n_ref, "chi": chi, "info": info_s, "seed": sd,
                              "value": e_bar * float(theta[0] + theta[1]) + float(theta[2]) * n_ref})
        if cands:
            cands.sort(key=lambda c: c["value"])
            best = cands[(len(cands) - 1) // 2]
            lab, k, theta, fit, table, level, need, B = (best[x] for x in ("lab", "k", "theta", "fit", "table", "level", "need", "B"))
            mask, big, n_ref, chi = (best[x] for x in ("mask", "big", "n_ref", "chi"))
            info.update(best["info"])
            info["seeds_with_signal"] = len(cands)
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
        if broad:
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
                lam = (level * Bc).clamp_min(1e-12)
                fits_background = (Xc < 0.5) | (torch.special.gammainc(Xc.clamp_min(1e-6), lam) >= broad_tail)  # P(Poisson(lam) >= X)
                tested = (Bc >= need) & (Xc <= level * 10.0**0.15 * Bc)
                rare = (Bc < need) & fits_background
                wide = (mask | (big.unsqueeze(1) & (rate_c <= delta * rmax_c.unsqueeze(0)) & (tested | rare))) & keep_c.unsqueeze(1)
                inc = ok & (aw > 0)
                if int(inc.sum()) >= 50:  # the wider set's expected counts, rescaled to what its cells show
                    scale = float(min(max(float(yw[inc].sum() / (muw[inc].sum() / scale).clamp_min(1e-9)), 0.25), 4.0))
            states[-1]["mask_broad"] = wide
            info["silent_genes_tested_median"] = float(torch.median(mask.sum(1).to(f64)))
            info["silent_genes_broad_median"] = float(torch.median(wide.sum(1).to(f64)))
            info["silent_counts_broad"] = float((Xc * wide.to(f64)).sum())
            info["cells_left_out_of_cluster_sums"] = int(odd.sum())
    return kappas, swaps, diag, states


@torch.no_grad()
def cell_multipliers(csr, rows, z, depth, unit, e, soup, chibar_p, state, called=None, doublet_fold: float = 5.0, doublet_share: float = 0.05):
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

    Doublet-like cells. A cell with more than doublet_fold times the counts background explains
    on its cluster's silent genes, and more than doublet_share of all its counts there, is
    returned in the flag vector and written to the output (glymph_doublet_like). Both conditions
    are needed: a nucleus in a dirty library has many counts on silent genes but no more than
    background explains, and a cell in a clean library can carry several times the usual
    background and still be a singlet. By default such a cell is still held to its cluster's
    silent genes, which strips a partner cell's exclusive genes from a true doublet. Sparing
    these cells (Config.spare_doublets) was tested on 2026-11-17 and is off: it did not reduce
    the over-removal on the one kidney library that has it (1.28 times the truth either way),
    and it left the most contaminated cells of BD Rhapsody and Parse mixtures uncleaned (removal
    91% to 88% and 82-90% to 67%). Users who call doublets after cleaning should drop flagged
    cells first or turn the option on.

    Returns (multiplier per row, diagnostics, doublet-like flag per row).
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
    diag["n_doublet_like"] = int(doublet_like.sum())
    if int(w.sum()) < 50:
        return k_n.to(torch.float32), diag, doublet_like
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
    return m.to(torch.float32), diag, doublet_like
