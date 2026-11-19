"""Multi-resolution silent sets, and a reproducible embedding to build them on.

A gene is silent in a group of cells when the group's counts for it are what background alone
explains. One resolution cannot serve two needs at once. Large groups have the counts to test a
gene, but they hide sub-groups that express it: the human immune cells of a mouse xenograft sat in
mouse clusters, every human gene was silent there, and Glymph 0.3.1 removed 92 to 99% of their
counts. Small groups are pure, but a rare gene has no counts to test in them.

So silence is decided on a tree. Every cluster of the calibration (the coarse level, where the
background level was fitted) is split into sub-clusters (the fine level).

What is used (since the first development round of 0.4.0, 2026-11-18): the cluster's silent genes
are found as before, with cells far above the fit left out of the sums, and a sub-cluster overrules
its cluster for a gene that a quarter or more of its own cells carry (subcluster_carriers). The
tree can only take genes out of a sub-cluster's silent set, never add any.

What was tried first and is kept as an option (tree_peel > 0, tree_silent below), for each gene:

  1. Isolated carriers. An entry far above what background explains for that cell (a Poisson test
     against several times the fitted background) marks a carrier. A group in which fewer than a
     few cells carry a gene is judged without them: one doublet must not make a gene "expressed"
     for the forty cells around it. This replaces leaving whole cells out of the cluster sums,
     which depended on the silent set it was used to build.
  2. Peeling. If the coarse cluster's counts exceed background, the sub-clusters with the largest
     excess are set aside, up to max_peel of them. If the rest is then at background, the gene is
     silent for the rest and expressed in the ones set aside: a gene that one sub-type expresses
     is still a negative control for its siblings, with the power of all of them pooled.
  3. Contradiction. A sub-cluster that holds far more of its parent's counts than its share of the
     background (Poisson test at the parent's observed level) expresses the gene even when the
     parent as a whole passes. That is the sub-group overruling its parent.

On the 23 development libraries that version lost removal where doublets are common (BD Rhapsody
91.6% to 88.3%: a few doublets per sub-cluster are too many to call isolated) and kept fewer marker
counts in their own cells on tissue (a gene one sub-type expresses weakly was made silent for it
once two stronger sub-types were set aside), so it is not the default.

The per-cell guard of 0.3.2 (background match and silent share of flagged cells) stays as a second
line for single cells that no sub-cluster represents.

The embedding. Up to 0.3.2 the clusters came from the model's latent space, which differs between
two runs of the same code on a GPU; on libraries with few testable genes the estimate moved with
it (cross-species removal 66.5% in one run and 87.4% in the next on one xenograft). pca_embedding
is a function of the counts and a seed only.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def pca_embedding(csr, fit_rows: torch.Tensor, rows: torch.Tensor, n_genes: int = 2000, n_comp: int = 30, seed: int = 0,
                  max_fit: int = 100_000, min_share: float = 0.005):
    """Principal components of log1p(counts per 10,000) on the most variable genes.

    The genes and the components are learned on fit_rows (at most max_fit of them, evenly spaced)
    and every row of `rows` is projected. Seeded randomised SVD, so the result depends on the data
    and the seed only. Returns float32 [len(rows), n_comp].
    """
    dev, G = csr.device, csr.shape[1]
    f64 = torch.float64
    if fit_rows.numel() > max_fit:
        fit_rows = fit_rows[torch.linspace(0, fit_rows.numel() - 1, max_fit, device=dev).long()]
    n = fit_rows.numel()

    def entries(rs):
        r, c, v, _ = csr.take(rs)
        tot = torch.zeros(rs.numel(), dtype=torch.float32, device=dev).index_add_(0, r, v)
        return r, c, torch.log1p(v / tot[r].clamp_min(1.0) * 1e4)

    s1, s2, nn = (torch.zeros(G, dtype=f64, device=dev) for _ in range(3))
    for a, b in csr.row_chunks(fit_rows):
        _, c, x = entries(fit_rows[a:b])
        x = x.to(f64)
        s1.index_add_(0, c, x)
        s2.index_add_(0, c, x * x)
        nn.index_add_(0, c, torch.ones_like(x))
    mean = s1 / max(n, 1)
    var = (s2 / max(n, 1) - mean * mean).clamp_min(0.0)
    ok = nn >= max(3.0, min_share * n)
    n_g = int(min(n_genes, int(ok.sum())))
    n_comp = int(min(n_comp, max(n_g - 1, 1), max(n - 1, 1)))
    genes = torch.topk(torch.where(ok, var, torch.full_like(var, -1.0)), n_g).indices.sort().values
    col = torch.full((G,), -1, dtype=torch.long, device=dev)
    col[genes] = torch.arange(n_g, device=dev)
    mu = mean[genes].to(torch.float32)

    def dense(rs):
        A = torch.zeros(rs.numel(), n_g, dtype=torch.float32, device=dev)
        for a, b in csr.row_chunks(rs):
            r, c, x = entries(rs[a:b])
            j = col[c]
            keep = j >= 0
            A[r[keep] + a, j[keep]] = x[keep]
        return A - mu

    A = dense(fit_rows)
    gen = torch.Generator().manual_seed(int(seed))
    Y = A @ torch.randn(n_g, n_comp + 10, generator=gen).to(dev)
    for _ in range(2):
        Q, _ = torch.linalg.qr(Y)
        Y = A @ (A.T @ Q)
    Q, _ = torch.linalg.qr(Y)
    _, _, Vh = torch.linalg.svd(Q.T @ A, full_matrices=False)
    V = Vh[:n_comp].T.contiguous()
    V = V * torch.sign(V[V.abs().argmax(0), torch.arange(V.shape[1], device=dev)]).unsqueeze(0)  # fixed signs
    del A, Y, Q
    out = torch.zeros(rows.numel(), n_comp, dtype=torch.float32, device=dev)
    step = 50_000
    for i in range(0, rows.numel(), step):
        out[i:i + step] = dense(rows[i:i + step]) @ V
    return out


def subclusters(z: torch.Tensor, lab: torch.Tensor, k: int, min_cells: int = 40, max_children: int = 8, seed: int = 0):
    """Split every cluster into up to max_children sub-clusters of about min_cells cells or more (k-means inside it).

    Returns (fine label per row, parent cluster of each fine cluster as a long tensor [F], centres [F, d] float64).
    A cluster without cells keeps one empty child whose centre is far from everything.
    """
    from .exclusion import kmeans

    dev = z.device
    fine = torch.zeros_like(lab)
    parent, centres, off = [], [], 0
    zd = z.to(torch.float64)
    for c in range(k):
        idx = torch.nonzero(lab == c, as_tuple=True)[0]
        m = int(min(max_children, idx.numel() // max(min_cells, 1)))
        if m >= 2:
            sub = torch.unique(kmeans(z[idx], m, seed=seed), return_inverse=True)[1]
            m = int(sub.max()) + 1
        else:
            sub, m = torch.zeros(idx.numel(), dtype=torch.long, device=dev), 1
        fine[idx] = off + sub
        for j in range(m):
            sel = idx[sub == j]
            centres.append(zd[sel].mean(0) if sel.numel() else torch.full((z.shape[1],), 1e30, dtype=torch.float64, device=dev))
        parent += [c] * m
        off += m
    return fine, torch.tensor(parent, dtype=torch.long, device=dev), torch.stack(centres)


@torch.no_grad()
def subcluster_carriers(csr, rows, fine, F: int, depth, unit, e, soup, chi, theta, n_ref, level, pos_fold: float = 3.0,
                        pos_tail: float = 1e-3, u=None, max_entries: int = 30_000_000):
    """Per sub-cluster and gene: counts, the number of cells that carry the gene, and unspliced molecules.

    A cell carries a gene when its entry is far above what background explains for that cell: at least two
    counts, and beyond the pos_tail tail of a Poisson with pos_fold times the fitted background as mean.
    Returns (X [F, G], carriers [F, G], unspliced [F, G] or None), float64.
    """
    dev, f64 = fine.device, torch.float64
    G = csr.shape[1]
    d = depth.to(f64)
    soup, chi = soup.to(f64), chi.to(f64)
    kn = float(theta[0]) + float(theta[1]) * d / n_ref
    eu = e.to(f64)[unit]
    X = torch.zeros(F * G, dtype=f64, device=dev)
    Npos = torch.zeros(F * G, dtype=f64, device=dev)
    Us = torch.zeros(F * G, dtype=f64, device=dev) if u is not None else None
    for a, b in csr.row_chunks(rows, max_entries):
        r, c, v, at = csr.take(rows[a:b])
        i = r + a
        lam = pos_fold * float(level) * (kn[i] * eu[i] * soup[unit[i], c] + float(theta[2]) * d[i] * chi[c])
        v = v.to(f64)
        key = fine[i] * G + c
        X.index_add_(0, key, v)
        pos = (v >= 2.0) & (torch.special.gammainc(v, lam.clamp_min(1e-12)) < pos_tail)
        Npos.index_add_(0, key[pos], torch.ones(int(pos.sum()), dtype=f64, device=dev))
        if u is not None:
            Us.index_add_(0, key, torch.minimum(u[at].to(f64), v))
        del r, c, v, at, i, lam, key, pos
    return X.view(F, G), Npos.view(F, G), (Us.view(F, G) if u is not None else None)


def _fits(X, B, lv, need, tail):
    """Counts X are what background lv * B explains: within 10**0.15 of it where enough is expected to test
    the ratio, inside the lower 1 - tail of the Poisson otherwise."""
    lam = (lv * B).clamp_min(1e-12)
    rare = (X < 0.5) | (torch.special.gammainc(X.clamp_min(1e-6), lam) >= tail)  # P(Poisson(lam) >= X)
    return torch.where(B >= need, X <= lv * 10.0**0.15 * B, rare)


@torch.no_grad()
def tree_silent(csr, rows, fine, parent, k, depth, unit, e, soup, chi, theta, n_ref, level, need, keep_c, big_c,
                delta: float = 0.2, tail: float = 0.05, pos_fold: float = 3.0, pos_tail: float = 1e-3, pos_min: int = 3,
                pos_frac: float = 0.08, conc_tail: float = 1e-3, max_peel: int = 2, max_entries: int = 30_000_000,
                splice: dict | None = None, splice_max_ratio: float = 4.0):
    """Silent genes of every fine cluster (see the module docstring).

    rows, fine, depth, unit: one entry per confident cell of the pool. parent [F]: coarse cluster of each
    fine cluster. theta = (kappa, kappa_d, s), level: the fitted background and the ridge level it is
    read against. keep_c, big_c [k]: coarse clusters the fit kept, and those with enough cells.
    splice: optional {"u": unspliced molecules per entry of csr.data, "pi_bg", "pi_nat", "informative"} (splice.py).
    With it a gene is also silent where its molecules are spliced like background (and no more than
    splice_max_ratio times the fitted background, beyond which a mismatch of the profile is not
    believable), and never where they are unspliced like the gene's own expression.
    Returns (silent by the counts [F, G] bool, silent by counts or splicing [F, G] bool: the set to strip,
    counts per unit depth [F, G] with isolated carriers left out, fitted background [F, G], diagnostics).
    """
    from .splice import splice_support, splice_veto

    dev, f64 = fine.device, torch.float64
    G, U = csr.shape[1], soup.shape[0]
    F = int(parent.numel())
    d = depth.to(f64)
    soup = soup.to(f64)
    chi = chi.to(f64)
    lv = float(level)
    kn = float(theta[0]) + float(theta[1]) * d / n_ref
    X = csr.group_sums(rows, fine, F, max_entries=max_entries)
    Xpos = torch.zeros(F * G, dtype=f64, device=dev)
    Npos = torch.zeros(F * G, dtype=f64, device=dev)
    Us = torch.zeros(F * G, dtype=f64, device=dev) if splice is not None else None      # unspliced molecules
    Uspos = torch.zeros(F * G, dtype=f64, device=dev) if splice is not None else None   # ... of the carriers
    eu = e.to(f64)[unit]
    for a, b in csr.row_chunks(rows, max_entries):
        r, c, v, at = csr.take(rows[a:b])
        i = r + a
        lam = pos_fold * lv * (kn[i] * eu[i] * soup[unit[i], c] + float(theta[2]) * d[i] * chi[c])
        v = v.to(f64)
        pos = (v >= 2.0) & (torch.special.gammainc(v, lam.clamp_min(1e-12)) < pos_tail)
        key = fine[i] * G + c
        idx = key[pos]
        Xpos.index_add_(0, idx, v[pos])
        Npos.index_add_(0, idx, torch.ones_like(idx, dtype=f64))
        if splice is not None:
            uu = torch.minimum(splice["u"][at].to(f64), v)
            Us.index_add_(0, key, uu)
            Uspos.index_add_(0, idx, uu[pos])
            del uu
        del r, c, v, i, lam, pos, idx, key, at
    Xpos, Npos = Xpos.view(F, G), Npos.view(F, G)
    n_f = torch.bincount(fine, minlength=F).to(f64)
    need_cells = torch.maximum(torch.full_like(n_f, float(pos_min)), torch.ceil(pos_frac * n_f))
    isolated = Npos < need_cells.unsqueeze(1)
    Xr = X - torch.where(isolated, Xpos, torch.zeros_like(Xpos))  # the group's counts without isolated carriers
    n_iso = float(torch.where(isolated, Npos, torch.zeros_like(Npos)).sum())
    Ur = None
    if splice is not None:
        Ur = Us.view(F, G) - torch.where(isolated, Uspos.view(F, G), torch.zeros_like(Xpos))
        pi_bg, pi_nat, inform = splice["pi_bg"].to(f64), splice["pi_nat"].to(f64), splice["informative"]
        del Us, Uspos
    del Xpos, Npos, isolated
    amb = e.to(f64).unsqueeze(1) * soup  # [U, G]
    one = torch.ones_like(d)
    n_fu = torch.zeros(F * U, dtype=f64, device=dev).index_add_(0, fine * U + unit, one).view(F, U)
    n2_fu = torch.zeros(F * U, dtype=f64, device=dev).index_add_(0, fine * U + unit, d / n_ref).view(F, U)
    N_f = torch.zeros(F, dtype=f64, device=dev).index_add_(0, fine, d)
    B = float(theta[0]) * (n_fu @ amb) + float(theta[1]) * (n2_fu @ amb) + float(theta[2]) * N_f.unsqueeze(1) * chi.unsqueeze(0)
    rate = Xr / N_f.clamp_min(1.0).unsqueeze(1)

    def up(M):  # sum of the children per coarse cluster
        return torch.zeros(k, M.shape[1], dtype=f64, device=dev).index_add_(0, parent, M)

    Xc, Bc = up(Xr), up(B)
    Uc = up(Ur) if splice is not None else None
    Nc = torch.zeros(k, dtype=f64, device=dev).index_add_(0, parent, N_f)
    # A gene's top rate: over coarse clusters with enough cells and over fine clusters with at least pos_min
    # cells' worth of evidence. A gene is a candidate where its rate is at most delta times that.
    rate_c = Xc / Nc.clamp_min(1.0).unsqueeze(1)
    rmax = torch.where(big_c.unsqueeze(1), rate_c, torch.zeros_like(rate_c)).max(0).values
    # A sub-cluster counts toward the top rate only where it clearly expresses the gene (several counts, three times
    # background or more): the rate of a few cells with one stray count is noise, and letting it set the top rate
    # would turn every rare gene into a candidate for silence in the rest of its cluster.
    clear = (Xr >= 5.0) & (Xr >= 3.0 * lv * B)
    rmax = torch.maximum(rmax, torch.where(clear, rate, torch.zeros_like(rate)).max(0).values)
    silent = torch.zeros(F, G, dtype=torch.bool, device=dev)
    by_counts = torch.zeros(F, G, dtype=torch.bool, device=dev)
    n_peeled = n_contra = n_support = n_veto = 0
    excess = Xr - lv * B
    for c in range(k):
        ch = torch.nonzero(parent == c, as_tuple=True)[0]
        if not bool(keep_c[c]) or not bool(big_c[c]) or ch.numel() == 0:
            continue
        m = ch.numel()
        p_max = min(max_peel, m - 1)
        top = torch.topk(excess[ch], p_max, dim=0).indices if p_max > 0 else torch.zeros(0, G, dtype=torch.long, device=dev)
        restX, restB, restN = Xc[c].clone(), Bc[c].clone(), torch.full((G,), float(Nc[c]), dtype=f64, device=dev)
        restU = Uc[c].clone() if splice is not None else None
        counted = torch.zeros(G, dtype=torch.bool, device=dev)  # the decision rests on the counts
        done = torch.zeros(G, dtype=torch.bool, device=dev)
        alive = torch.ones(G, dtype=torch.bool, device=dev)  # every sub-cluster set aside so far clearly expresses the gene
        n_set_aside = torch.zeros(G, dtype=torch.long, device=dev)
        final = {}
        g = torch.arange(G, device=dev)
        for p in range(p_max + 1):
            cnt = _fits(restX, restB, lv, need, tail) & (restX / restN.clamp_min(1.0) <= delta * rmax)
            ok = cnt
            if splice is not None:
                sup = splice_support(restX, restU, pi_bg, pi_nat, inform) & (restX <= splice_max_ratio * lv * restB)
                veto = splice_veto(restX, restU, pi_bg, pi_nat, inform)
                ok = (cnt | sup) & ~veto
                n_support += int((sup & ~cnt & ~veto & ~done & alive).sum())
                n_veto += int((cnt & veto & ~done & alive).sum())
            ok = ok & ~done & alive
            final[p] = (ok, restX.clone(), restB.clone())
            n_set_aside[ok] = p
            counted |= ok & cnt
            done |= ok
            if p < p_max:
                # Set the sub-cluster with the next largest excess aside, but only if it clearly expresses the gene.
                # A cluster that is a little above background everywhere is not made silent by dropping its top part.
                j = ch[top[p]]
                xj, bj = Xr[j, g], B[j, g]
                alive &= (xj >= 3.0) & (torch.special.gammainc(xj.clamp_min(1e-6), (3.0 * lv * bj).clamp_min(1e-12)) < pos_tail)
                restX = restX - xj
                restB = restB - bj
                restN = restN - N_f[j]
                if splice is not None:
                    restU = restU - Ur[j, g]
        if p_max > 0:
            rank = torch.empty(m, G, dtype=torch.long, device=dev).fill_(m)  # position of each child in the excess order
            rank.scatter_(0, top, torch.arange(p_max, device=dev).unsqueeze(1).expand(p_max, G))
        else:
            rank = torch.full((m, G), m, dtype=torch.long, device=dev)
        aside = rank < n_set_aside.unsqueeze(0)  # children set aside for this gene
        # the level the rest shows, as the yardstick for a child that holds more than its share
        rx = torch.zeros(G, dtype=f64, device=dev)
        rb = torch.ones(G, dtype=f64, device=dev)
        for p, (ok, x_, b_) in final.items():
            rx[ok], rb[ok] = x_[ok], b_[ok]
        lam = torch.maximum(rx.unsqueeze(0) * B[ch] / rb.clamp_min(1e-12).unsqueeze(0), lv * B[ch]).clamp_min(1e-12)
        contra = (Xr[ch] >= 3.0) & (torch.special.gammainc(Xr[ch].clamp_min(1e-6), lam) < conc_tail)
        if splice is not None:  # a sub-cluster whose own molecules of the gene are unspliced like expression
            contra |= splice_veto(Xr[ch], Ur[ch], pi_bg.unsqueeze(0), pi_nat.unsqueeze(0), inform.unsqueeze(0))
        s = done.unsqueeze(0) & ~aside & ~contra
        silent[ch] = s
        by_counts[ch] = s & counted.unsqueeze(0)
        n_peeled += int((done & (n_set_aside > 0)).sum())
        n_contra += int((done.unsqueeze(0) & ~aside & contra).sum())
    diag = {"fine_clusters": F, "fine_cells_median": float(torch.median(n_f)), "isolated_carrier_entries": int(n_iso),
            "silent_counts": float((Xr * by_counts.to(f64)).sum()),
            "genes_silent_after_setting_subclusters_aside": n_peeled, "subcluster_gene_pairs_overruling_parent": n_contra}
    if splice is not None:
        diag.update({"cluster_gene_pairs_silent_by_splicing_only": n_support, "cluster_gene_pairs_vetoed_by_splicing": n_veto,
                     "counts_silent_by_splicing_only": float((Xr * (silent & ~by_counts).to(f64)).sum())})
    return by_counts, silent, rate, B, diag


@torch.no_grad()
def shared_level(X, B, cand, need: float, tail: float = 0.05, factor_max: float = 3.0, min_clusters: int = 2,
                 min_expected: float = 10.0, window: float = 10.0**0.15, prior: float = 3.0):
    """Silent at a level that the clusters share: the gene's own background level, read across clusters.

    The fitted background gives every gene the soup's share (and the average cell's, for the swapped part). Gene by
    gene the background inside cells departs from that: in cell-line mixtures, where the other species' genes are
    background by construction, their counts sit at 0.6 to 1.8 times the fitted value (10th to 90th percentile,
    2026-11-18), and a gene at 1.5 to 3 times it failed the silent test in every cluster and kept 20 to 30% of
    its background. Expression cannot be told from such a gene in one cluster. Across clusters it can: the
    background's profile is one for the library, so a gene that is only background sits at the same multiple
    of the fit in every cluster that does not express it, and expression lifts some clusters above the rest.

    X, B [k, G]: observed counts and fitted background (already at the test level). cand [k, G]: pairs where
    the cluster's rate is at most a fifth of the top cluster's. Per gene the candidate clusters are ordered by
    observed over fitted, and from the lowest upward a cluster joins while its counts agree with the pooled
    multiple of those before it (within the window of the silent test where enough counts are expected, the
    exact Poisson test where not). The pooled multiple is drawn to the fit while few counts are expected (prior
    counts at multiple 1), so a rare gene stays where the plain test puts it. The gene has a shared level when min_clusters or more clusters
    agree on a multiple between 1 / factor_max and factor_max, with min_expected counts expected among them.
    A multiple below 1 changes nothing for the silent test (such a gene passes it anyway); it is returned because
    it is the gene's background level in the cells that do express it, too (mitochondrial genes on BD Rhapsody
    sit at 0.5 to 0.7 of the fit in every cluster of the other species).

    Returns (member [k, G] bool: the pairs silent at the shared level; factor [G]: the multiple, 1 where none).
    """
    f64 = torch.float64
    k, G = X.shape
    Xc, Bc = (X * cand).to(f64), (B * cand).to(f64)
    r = torch.where(cand & (B > 0), X / B.clamp_min(1e-12), torch.full_like(X, float("inf")))
    order = r.argsort(0)
    Xs, Bs, cs = Xc.gather(0, order), Bc.gather(0, order), (cand & (B > 0)).gather(0, order)
    cx, cb = Xs.cumsum(0), Bs.cumsum(0)
    pooled = (cx + prior) / (cb + prior)  # multiple of the first j + 1 clusters, drawn to 1 while few counts are expected
    # the multiple of the clusters before this one; the lowest cluster has none before it and starts the group
    # wherever it lies, up to the largest multiple allowed
    first = torch.full((1, G), float(factor_max), dtype=f64, device=X.device)
    before = torch.cat([first, pooled[:-1]])
    lam = (before.clamp_min(1.0) * Bs).clamp_min(1e-12)
    ok_big = Xs <= window * lam
    ok_rare = (Xs < 0.5) | (torch.special.gammainc(Xs.clamp_min(1e-6), lam) >= tail)
    join = cs & torch.where(lam >= need, ok_big, ok_rare)
    chain = join.to(torch.int64).cumprod(0).bool()
    L = chain.sum(0)
    idx = (L - 1).clamp_min(0).unsqueeze(0)
    factor = pooled.gather(0, idx)[0]
    good = (L >= min_clusters) & (cb.gather(0, idx)[0] >= min_expected) & (factor <= factor_max)
    member = torch.zeros_like(cand).scatter(0, order, chain) & good.unsqueeze(0)
    return member, torch.where(good, factor.clamp_min(1.0 / factor_max), torch.ones_like(factor))

