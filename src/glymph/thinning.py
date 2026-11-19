"""Integer output by thinning the observed counts.

Each nonzero entry x has a posterior background share p (from the mixture rates), so
the number of background counts in it is Binomial(x, p). Every estimator below removes
between 0 and x counts from an entry and never adds any, so the denoised matrix is
bounded by the raw matrix by construction.

Estimators
  gene      (default since 0.1.0) per gene, remove round(sum over cells of the expected
            background counts) and give them to the entries where background is most
            probable. See gene_budget below. Gene totals match the model's expectation;
            within a gene the removal goes first to cells where the gene is otherwise
            silent, which is where a background count is most likely to be sitting.
  budget    per barcode, remove round(sum of expected background counts) and
            give them to entries by largest-remainder apportionment. Deterministic,
            integer, and matches each barcode's expected background total. Remainders
            of large native entries compete with single background counts, so on real
            mixtures it left about a third of the known background in (the v0 default).
  sample    one binomial draw per entry. Unbiased per gene and per cell, but random.
  map       per-entry posterior mode, floor((x + 1) * p).
  expected  x * p, not integer. For diagnostics only.

strength scales p before the estimator (1.0 = the model's own estimate).
"""

from __future__ import annotations

import torch


def apportion(r, v, share, n_rows: int, mode: str = "budget", strength: float = 1.0):
    p = (share * strength).clamp(0.0, 1.0)
    expected = v * p
    if mode == "expected":
        return expected
    if mode == "sample":
        dev = v.device
        cpu = dev.type == "mps"  # torch.binomial is not implemented on MPS
        m = torch.binomial(v.cpu() if cpu else v, p.cpu() if cpu else p)
        return m.to(dev)
    if mode == "map":
        return torch.minimum(torch.floor((v + 1.0) * p), v)
    if mode != "budget":
        raise ValueError(mode)

    base = torch.floor(expected)
    frac = expected - base
    zeros = torch.zeros(n_rows, device=v.device, dtype=v.dtype)
    deficit = torch.round(zeros.index_add(0, r, expected)) - zeros.index_add(0, r, base)
    # Order entries by barcode, and within a barcode by descending remainder.
    by_frac = torch.argsort(frac, descending=True)
    order = by_frac[torch.argsort(r[by_frac], stable=True)]
    r_sorted = r[order]
    counts = torch.bincount(r, minlength=n_rows)
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(r.numel(), device=v.device) - first[r_sorted]
    take = (rank < deficit[r_sorted]) & (frac[order] > 0)
    extra = torch.zeros_like(v)
    extra[order[take]] = 1.0
    return torch.minimum(base + extra, v)


def _allowance(g, v, p, n_genes: int, fpr: float, cap):
    """Per-entry false-positive allowance: fpr of the entry's native counts, scaled per gene so that a gene's
    allowance never exceeds cap times its expected background (cap None: no limit)."""
    extra = fpr * v * (1.0 - p)
    if cap is None or fpr <= 0:
        return extra
    z = torch.zeros(n_genes, device=v.device, dtype=torch.float64)
    bg = z.index_add(0, g, (v * p).double())
    ex = z.index_add(0, g, extra.double())
    factor = torch.where(ex > 0, (cap * bg / ex.clamp_min(1e-12)).clamp(max=1.0), torch.ones_like(ex))
    return extra * factor[g].to(extra.dtype)


def gene_budget(g, v, share, n_genes: int, strength: float = 1.0, fpr: float = 0.0, iters: int = 44, cap=None):
    """Per-gene background budget, allocated to the most probable entries.

    g, v, share: gene index, count and posterior background share of every nonzero entry of
    the called cells. For each gene the number of counts removed is
    R_g = round(sum over cells of v * p). Which entries give them up is the allocation that
    maximises the posterior probability: if the background count in an entry is
    Binomial(v, p), the gain from removing one more count from an entry that has already
    given c is ((v - c) / (c + 1)) * p / (1 - p), which falls as c grows. Taking counts in
    order of that gain until the gene's budget is spent is a water-filling: with threshold t,

        c(t) = min(v, floor((v + 1) * o / (o + t))),   o = p / (1 - p)

    and t is found per gene by bisection so that the total meets R_g. Counts left over after
    the bisection (entries tied at the threshold) go one per entry in order of gain. The
    result is integer, never exceeds v, and reproduces each gene's expected background total.
    This is the same objective as CellBender's knapsack output step, solved in closed form
    plus a bisection instead of a solver.

    fpr adds a false-positive budget, as CellBender's --fpr does (its default is 0.01): each
    gene's budget becomes R_g = round(sum of v * p + fpr * sum of v * (1 - p)), that is, up to
    a share fpr of the gene's native counts may be removed on top of the expected background.
    The extra counts are taken the same way, from the entries where background is most
    probable, so they go first to single counts in cells that barely express the gene. Use 0
    to remove exactly the expected background.

    cap limits that allowance to a share of the gene's expected background. A share of native
    counts is small beside the background of a dirty library and large beside that of a clean
    one: at 1% background, an allowance of 0.4% of native counts is 40% more than the
    background itself (xenograft skin, 2026-11-17: twice the expected same-species background
    removed). With cap = 0.2 a gene never gives up more than 1.2 times its expected background,
    and a gene with no expected background gives up nothing.
    """
    p = (share * strength).clamp(1e-6, 1.0 - 1e-6)  # float32-safe: 1 - 1e-9 would round to 1 and give infinite odds
    o = p / (1.0 - p)
    zeros = torch.zeros(n_genes, device=v.device, dtype=torch.float64)
    target = torch.round(zeros.index_add(0, g, (v * p + _allowance(g, v, p, n_genes, fpr, cap)).double()))
    lo = torch.full((n_genes,), -40.0, device=v.device)  # log t: everything removed
    hi = torch.full((n_genes,), 40.0, device=v.device)  # log t: nothing removed
    vp1 = v + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = torch.minimum(torch.floor(vp1 * o / (o + torch.exp(mid)[g])), v)
        over = zeros.index_add(0, g, c.double()) > target
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    c = torch.minimum(torch.floor(vp1 * o / (o + torch.exp(hi)[g])), v)
    deficit = target - zeros.index_add(0, g, c.double())
    gain = torch.where(c < v, (v - c) / (c + 1.0) * o, torch.full_like(v, -1.0))
    by_gain = torch.argsort(gain, descending=True)
    order = by_gain[torch.argsort(g[by_gain], stable=True)]
    g_sorted = g[order]
    counts = torch.bincount(g, minlength=n_genes)
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(g.numel(), device=v.device) - first[g_sorted]
    take = (rank < deficit[g_sorted]) & (gain[order] > 0)
    extra = torch.zeros_like(v)
    extra[order[take]] = 1.0
    return torch.minimum(c + extra, v)


def gene_budget_blocked(g, v, share, n_genes: int, strength: float = 1.0, fpr: float = 0.0, max_entries: int = 20_000_000, cap=None):
    """gene_budget over blocks of genes, so peak memory does not grow with the library.

    Each gene's budget is independent of every other gene's, so the allocation can be done for a
    few thousand genes at a time. A library of 2 million barcodes (Parse, 300 million entries)
    ran out of memory on a 24 GB card in the single-pass version; the result is identical.
    If a block still does not fit, it is done on the CPU.
    """
    if g.numel() <= max_entries:
        return gene_budget(g, v, share, n_genes, strength, fpr, cap=cap)
    counts = torch.bincount(g, minlength=n_genes)
    edges, acc, start = [0], 0, 0
    for j, n in enumerate(counts.tolist()):
        if acc > 0 and acc + n > max_entries:
            edges.append(j)
            acc = 0
        acc += n
    edges.append(n_genes)
    out = torch.zeros_like(v)
    for g0, g1 in zip(edges[:-1], edges[1:]):
        idx = torch.nonzero((g >= g0) & (g < g1), as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        try:
            out[idx] = gene_budget(g[idx] - g0, v[idx], share[idx], g1 - g0, strength, fpr, cap=cap)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            out[idx] = gene_budget((g[idx] - g0).cpu(), v[idx].cpu(), share[idx].cpu(), g1 - g0, strength, fpr, cap=cap).to(v.device)
        del idx
    return out


def gene_cell_budget(g, r, v, share, n_genes: int, n_cells: int, strength: float = 1.0, fpr: float = 0.0,
                     outer: int = 3, iters: int = 40, cap=None):
    """Gene budget that also keeps each cell's own background total.

    gene_budget fixes how many counts each gene gives up across the library and takes them from
    the entries where background is most probable. Those are mostly in shallow cells, so the
    per-cell totals drift: on the kidney truth set shallow cells lost 1.0 to 1.2 times their
    background and deep cells 0.5 to 0.7 times, although the posterior put about the same number
    of background counts in each. Here every entry's odds are scaled by a factor per cell as well
    as a threshold per gene,

        c = min(v, floor((v + 1) * sigmoid(logit(p) + a_cell - b_gene)))

    and the two sets of factors are found in turn by bisection so that each gene meets its budget
    and each cell its expected background (counts with the same false-positive allowance). The
    last pass is over genes, so gene totals are exact as in gene_budget (ties broken by gain) and
    cell totals are met to within a few counts. Integer, never above v.
    """
    p = (share * strength).clamp(1e-6, 1.0 - 1e-6)
    logit = torch.log(p) - torch.log1p(-p)
    w = (v * p + _allowance(g, v, p, n_genes, fpr, cap)).double()  # allowance capped per gene, see gene_budget
    zg = torch.zeros(n_genes, device=v.device, dtype=torch.float64)
    zc = torch.zeros(n_cells, device=v.device, dtype=torch.float64)
    target_g = torch.round(zg.index_add(0, g, w))
    target_c = zc.index_add(0, r, w)
    a = torch.zeros(n_cells, device=v.device)
    b = torch.zeros(n_genes, device=v.device)
    vp1 = v + 1.0

    def counts(a_, b_):
        return torch.minimum(torch.floor(vp1 * torch.sigmoid(logit + a_[r] - b_[g])), v)

    def gene_pass():
        lo = torch.full((n_genes,), -40.0, device=v.device)  # everything removed
        hi = torch.full((n_genes,), 40.0, device=v.device)  # nothing removed
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            over = zg.index_add(0, g, counts(a, mid).double()) > target_g
            lo = torch.where(over, mid, lo)
            hi = torch.where(over, hi, mid)
        return hi

    def cell_pass():
        lo = torch.full((n_cells,), -40.0, device=v.device)  # nothing removed
        hi = torch.full((n_cells,), 40.0, device=v.device)  # everything removed
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            over = zc.index_add(0, r, counts(mid, b).double()) > target_c
            hi = torch.where(over, mid, hi)
            lo = torch.where(over, lo, mid)
        return lo

    b = gene_pass()
    for _ in range(outer):
        a = cell_pass().clamp(-8.0, 8.0)  # a cell's factor may tilt the odds, not override them
        b = gene_pass()
    c = counts(a, b)
    deficit = target_g - zg.index_add(0, g, c.double())
    odds = torch.exp((logit + a[r] - b[g]).clamp(-30.0, 30.0))
    gain = torch.where(c < v, (v - c) / (c + 1.0) * odds, torch.full_like(v, -1.0))
    by_gain = torch.argsort(gain, descending=True)
    order = by_gain[torch.argsort(g[by_gain], stable=True)]
    g_sorted = g[order]
    n_per = torch.bincount(g, minlength=n_genes)
    first = torch.cumsum(n_per, 0) - n_per
    rank = torch.arange(g.numel(), device=v.device) - first[g_sorted]
    take = (rank < deficit[g_sorted]) & (gain[order] > 0)
    extra = torch.zeros_like(v)
    extra[order[take]] = 1.0
    return torch.minimum(c + extra, v)
