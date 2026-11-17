#!/usr/bin/env python
"""The label-free steps added for tissue, BD Rhapsody and split-pool data.

  1. silent_budget recovers a known ambient multiplier and swapped fraction from four cell types,
     with 5% doublets that express the "silent" genes for real.
  2. estimate_priors picks the plateau on a droplet-like depth curve, the knee on a curve with a
     long tail and no plateau, and refuses a plateau made of two kinds of cells.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from glymph.background import estimate_priors  # noqa: E402
from glymph.exclusion import silent_budget  # noqa: E402
from glymph.sparse import DeviceCSR  # noqa: E402


def tissue(kappa=0.8, s=0.03, e=200.0, n=6000, seed=0, doublets=0.05):
    rng = np.random.default_rng(seed)
    G, K, spec = 2000, 4, 300
    prof = np.zeros((K, G))
    shared = rng.gamma(0.5, 1.0, G - K * spec)
    for k in range(K):
        prof[k, K * spec:] = 0.6 * shared / shared.sum() * rng.lognormal(0, 0.2, shared.size)
        own = rng.gamma(0.5, 1.0, spec)
        prof[k, k * spec:(k + 1) * spec] = 0.4 * own / own.sum()
        prof[k] /= prof[k].sum()
    typ = rng.choice(K, n, p=[0.6, 0.2, 0.1, 0.1])
    depth = rng.lognormal(np.log(3000), 0.5, n)
    chibar = (prof[typ] * depth[:, None]).sum(0)
    chibar /= chibar.sum()
    soup = chibar * rng.lognormal(0, 0.5, G)
    soup /= soup.sum()
    X = rng.poisson(prof[typ] * depth[:, None]).astype(np.float64)
    dbl = rng.random(n) < doublets
    other = (typ + rng.integers(1, K, n)) % K
    X[dbl] += rng.poisson(prof[other[dbl]] * (0.5 * depth[dbl])[:, None])
    native = X.sum(1)
    eps = rng.gamma(10.0, 0.1, n)
    X += rng.poisson(kappa * e * eps[:, None] * soup[None, :])
    X += rng.poisson(s * native[:, None] * chibar[None, :])
    z = np.eye(K)[typ] + rng.normal(0, 0.1, (n, K))
    return sp.csr_matrix(X), z, soup, chibar


def check_budget():
    out = []
    for kappa, s in ((0.8, 0.03), (1.5, 0.005), (0.4, 0.06)):
        X, z, soup, chibar = tissue(kappa, s)
        csr = DeviceCSR(X.astype(np.float32), torch.device("cpu"))
        n = X.shape[0]
        zero = torch.zeros(n, dtype=torch.long)
        k_hat, s_hat, diag, _ = silent_budget(
            csr, torch.arange(n), torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(np.asarray(X.sum(1)).ravel()),
            zero, zero, torch.tensor([200.0]), torch.as_tensor(soup, dtype=torch.float32)[None, :],
            torch.as_tensor(chibar, dtype=torch.float32)[None, :], torch.tensor([0.01]), n_clusters=8)
        assert k_hat[0] is not None, diag
        N = np.asarray(X.sum(1)).ravel()
        d0 = diag["0"]
        ratios = []
        for q in (0.1, 0.5, 0.9):  # the fitted background per cell must match across the depth range
            Nq = float(np.quantile(N, q))
            est = (d0["kappa"] + d0["kappa_depth"] * Nq / d0["depth_ref"]) * 200 + d0["s"] * Nq
            ratios.append(round(est / (kappa * 200 + s * Nq), 3))
        out.append((kappa, s, round(d0["kappa"], 3), round(d0["kappa_depth"], 3), round(d0["s"], 4), ratios, d0["cells_trimmed"]))
        # Within 20%: where the soup resembles the average cell, the fit puts depth-proportional
        # background on the ambient-profile term whatever its true source, and reads it 10 to 16% low.
        assert all(abs(r - 1) < 0.2 for r in ratios), f"background per cell off: {out[-1]}"
    print("silent_budget (kappa, s true | kappa, kappa_depth, s fitted | fitted/true background at depth q10, q50, q90 | cells trimmed):", out)


def check_modes():
    rng = np.random.default_rng(1)
    G = 400
    p1, p2 = rng.dirichlet(np.ones(G) * 0.3), rng.dirichlet(np.ones(G) * 0.3)

    def lib(depths, profiles):
        return sp.csr_matrix(np.vstack([rng.multinomial(int(d), p) for d, p in zip(depths, profiles)]))

    # droplet-like: 600 cells, a plateau of 6,000 cell-free partitions at 300, and a junk tail
    d_cells, d_emp = rng.lognormal(np.log(8000), 0.4, 600), rng.lognormal(np.log(300), 0.12, 6000)
    d_junk = np.minimum(5 * (1 - rng.random(8000)) ** (-1 / 1.5), 100)
    soup = 0.5 * (p1 + p2)
    X = lib(np.r_[d_cells, d_emp, d_junk], [p1 if i % 2 else p2 for i in range(600)] + [soup] * 14000)
    d = estimate_priors(X).diagnostics
    assert d["threshold_method"] == "plateau" and 250 < d["empty_level"] < 360, d
    assert d["confident_threshold"] > 400 and abs(d["n_confident"] - 600) < 30, d
    # knee and tail, no plateau: 800 cells of two kinds, then a power-law tail
    d_cells = rng.lognormal(np.log(3000), 0.35, 800)
    d_tail = 5 * (1 - rng.random(20000)) ** (-1 / 0.8)
    d_tail = d_tail[d_tail < 250]
    X = lib(np.r_[d_cells, d_tail], [p1 if i % 2 else p2 for i in range(800)] + [soup] * d_tail.size)
    d2 = estimate_priors(X).diagnostics
    assert d2["threshold_method"] == "knee", d2
    assert 250 <= d2["cell_threshold"] < 1500 and abs(d2["n_cells_prior"] - 800) < 60, d2
    print("modes ok:", d["threshold_method"], round(d["empty_level"]), round(d["confident_threshold"]), "|",
          d2["threshold_method"], round(d2["cell_threshold"]), d2["n_cells_prior"], round(d2["empty_level"]),
          "rejected:", d2["rejected_plateau"])




def check_blocked_budget():
    from glymph.thinning import gene_budget, gene_budget_blocked

    gen = torch.Generator().manual_seed(0)
    n, G = 200_000, 500
    g = torch.randint(G, (n,), generator=gen)
    v = torch.poisson(torch.rand(n, generator=gen) * 5).clamp_min(1.0)
    share = torch.rand(n, generator=gen)
    a = gene_budget(g, v, share, G, 1.0, 0.005)
    b = gene_budget_blocked(g, v, share, G, 1.0, 0.005, max_entries=30_000)
    assert torch.equal(a, b), "blocked gene budget differs from the single pass"
    print("blocked gene budget identical to the single pass")
    from glymph.thinning import gene_cell_budget

    n_cells = 3000
    r = torch.randint(n_cells, (n,), generator=gen)
    c = gene_cell_budget(g, r, v, share, G, n_cells, 1.0, 0.005)
    w = (v * (share.clamp(1e-6, 1 - 1e-6) + 0.005 * (1 - share.clamp(1e-6, 1 - 1e-6)))).double()
    tg = torch.round(torch.zeros(G, dtype=torch.float64).index_add(0, g, w))
    tc = torch.zeros(n_cells, dtype=torch.float64).index_add(0, r, w)
    got_g = torch.zeros(G, dtype=torch.float64).index_add(0, g, c.double())
    got_c = torch.zeros(n_cells, dtype=torch.float64).index_add(0, r, c.double())
    got_c0 = torch.zeros(n_cells, dtype=torch.float64).index_add(0, r, a.double())
    assert bool((c <= v).all()) and bool((c >= 0).all()) and bool((c == c.round()).all())
    assert float((got_g - tg).abs().max()) <= 1.0, "gene totals not met"
    err, err0 = float((got_c - tc).abs().mean()), float((got_c0 - tc).abs().mean())
    assert err < 0.5 * err0, f"cell totals not improved: {err} vs {err0}"
    print(f"gene + cell budget: gene totals exact, mean cell total error {err:.2f} counts (gene-only rule {err0:.2f})")


if __name__ == "__main__":
    check_modes()
    check_budget()
    check_blocked_budget()
