"""Smoke test on a small synthetic library. Runs on CPU in about a minute.

    python tests/test_smoke.py            # prints the metrics
    pytest tests/test_smoke.py

It checks that the code is wired correctly and that the integrity guarantees hold.
It is not evidence of accuracy on real data.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from glymph import Config, remove_background  # noqa: E402
from glymph.simulate import simulate  # noqa: E402
from glymph.exclusion import exclusion_budget, exclusion_swap_fraction  # noqa: E402
from glymph.sparse import DeviceCSR  # noqa: E402
from glymph.thinning import apportion, gene_budget  # noqa: E402


def run(estimator="gene", likelihood="multinomial", seed=0):
    torch.set_num_threads(4)
    raw, truth = simulate(seed=seed)
    cfg = Config(target_steps=800, min_epochs=10, estimator=estimator, likelihood=likelihood, device="cpu", seed=seed)
    out, rep = remove_background(raw, config=cfg, log=lambda *a: None)

    den = out.X.toarray()
    rawc = out.layers["raw"].toarray()
    true_cells = truth["native"].index
    called = out.obs_names
    hit = called.intersection(true_cells)
    kinds = raw.obs["kind"]
    idx = [out.obs_names.get_loc(b) for b in hit]
    nat = truth["native"].loc[hit].to_numpy()
    bkg = truth["background"].loc[hit].to_numpy()
    removed = (rawc - den)[idx]
    return {
        "estimator": estimator,
        "likelihood": likelihood,
        "max_above_raw": int((den - rawc).max()),
        "min_value": int(den.min()),
        "integer": bool(np.array_equal(den, np.round(den))),
        "cell_recall": len(hit) / len(true_cells),
        "false_cells": int((kinds.loc[called] != "cell").sum()),
        "background_removed": float(removed.sum() / bkg.sum()),
        "native_retained": float(1.0 - np.maximum(removed - bkg, 0).sum() / nat.sum()),
        "e_learned": list(rep["e_unit"].values())[0],
        "e_true": truth["ambient_level"],
        "s_learned": list(rep["swap_fraction"].values())[0],
        "s_true": truth["swap_fraction"],
        "train_s": rep["timings"]["train_s"],
        "epochs": rep["train"]["epochs_run"],
    }


def test_apportion_bounds():
    g = torch.Generator().manual_seed(0)
    r = torch.sort(torch.randint(0, 50, (5000,), generator=g)).values
    v = torch.randint(1, 20, (5000,), generator=g).float()
    p = torch.rand(5000, generator=g)
    for mode in ("budget", "sample", "map"):
        m = apportion(r, v, p, 50, mode)
        assert bool(((m >= 0) & (m <= v)).all()), mode
        assert bool((m == torch.round(m)).all()), mode
    m = apportion(r, v, p, 50, "budget")
    want = torch.round(torch.zeros(50).index_add(0, r, v * p))
    got = torch.zeros(50).index_add(0, r, m)
    assert bool((want == got).all())


def test_gene_budget():
    """Integer, within [0, v], and every gene's total equals its rounded expected background."""
    gen = torch.Generator().manual_seed(1)
    n, n_genes = 20000, 300
    g = torch.randint(0, n_genes, (n,), generator=gen)
    v = torch.randint(1, 40, (n,), generator=gen).float()
    v[:2000] = 1.0  # plenty of single counts, as in real data
    p = torch.rand(n, generator=gen) ** 3  # mostly small shares, a few near 1
    p[:50], p[50:100] = 1.0, 0.0  # exact 0 and 1 must not produce NaN
    m = gene_budget(g, v, p, n_genes)
    assert bool(torch.isfinite(m).all())
    assert bool(((m >= 0) & (m <= v)).all())
    assert bool((m == torch.round(m)).all())
    pc = p.clamp(1e-6, 1.0 - 1e-6)
    want = torch.round(torch.zeros(n_genes, dtype=torch.float64).index_add(0, g, (v * pc).double()))
    got = torch.zeros(n_genes, dtype=torch.float64).index_add(0, g, m.double())
    assert bool((want == got).all()), (want - got).abs().max()
    # a false-positive budget removes more, by about fpr times the native counts, never above v
    m2 = gene_budget(g, v, p, n_genes, fpr=0.05)
    assert bool(((m2 >= 0) & (m2 <= v)).all()) and bool((m2 == torch.round(m2)).all())
    extra = float(m2.sum() - m.sum())
    native = float((v * (1 - pc)).sum())
    assert 0.03 * native < extra < 0.07 * native, (extra, native)
    # removal goes to the most probable entries: among single counts of one gene, a removed one
    # never has a lower share than a kept one
    for gene in range(5):
        sel = (g == gene) & (v == 1)
        if sel.sum() > 1 and m[sel].sum() > 0 and (m[sel] == 0).any():
            assert p[sel][m[sel] == 1].min() >= p[sel][m[sel] == 0].max() - 1e-6


def test_exclusion_bound():
    """Two populations with disjoint gene sets, known ambient level and swap fraction."""
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n, G, depth, e_true, s_true = 2400, 400, 20000, 200.0, 0.02
    kind = rng.integers(0, 2, n)
    prof = np.zeros((2, G))
    prof[0, : G // 2] = rng.dirichlet(np.full(G // 2, 2.0))
    prof[1, G // 2 :] = rng.dirichlet(np.full(G // 2, 2.0))
    N = rng.lognormal(np.log(depth), 0.3, n)
    chibar = (prof[kind] * N[:, None]).sum(0)
    chibar /= chibar.sum()
    soup = 0.5 * prof[0] + 0.5 * prof[1]
    rate = prof[kind] * N[:, None] * (1 - s_true) + e_true * soup[None, :] + s_true * N[:, None] * chibar[None, :]
    X = sp.csr_matrix(rng.poisson(rate).astype(np.float32))
    csr = DeviceCSR(X, torch.device("cpu"))
    z = torch.tensor(np.eye(2)[kind] + 0.05 * rng.standard_normal((n, 2)), dtype=torch.float32)
    zero = torch.zeros(n, dtype=torch.long)
    est, diag = exclusion_swap_fraction(
        csr, torch.arange(n), z, zero, zero, torch.tensor([e_true]), torch.tensor(soup[None, :], dtype=torch.float32),
        torch.tensor(chibar[None, :], dtype=torch.float32),
    )
    assert est[0] is not None, diag
    assert 0.7 * s_true < est[0] < 1.3 * s_true, (est, diag)
    # one homogeneous population has no silent genes: no estimate, the caller keeps the learned value
    rate1 = prof[0][None, :] * N[:, None]
    csr1 = DeviceCSR(sp.csr_matrix(rng.poisson(rate1).astype(np.float32)), torch.device("cpu"))
    est1, _ = exclusion_swap_fraction(
        csr1, torch.arange(n), torch.tensor(0.05 * rng.standard_normal((n, 2)), dtype=torch.float32), zero, zero,
        torch.tensor([0.0]), torch.tensor(prof[0][None, :], dtype=torch.float32), torch.tensor(prof[0][None, :], dtype=torch.float32),
    )
    assert est1[0] is None, est1


def test_exclusion_budget():
    """Two populations, constant ambient plus depth-proportional swapping: recover both."""
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    n, G, e_true, kappa_true, s_true = 3600, 400, 200.0, 1.3, 0.02
    kind = rng.integers(0, 2, n)
    prof = np.zeros((2, G))
    prof[0, : G // 2] = rng.dirichlet(np.full(G // 2, 2.0))
    prof[1, G // 2 :] = rng.dirichlet(np.full(G // 2, 2.0))
    N = rng.lognormal(np.log(15000), 0.6, n)  # wide depth range, so strata differ
    chibar = (prof[kind] * N[:, None]).sum(0)
    chibar /= chibar.sum()
    soup = 0.5 * prof[0] + 0.5 * prof[1]
    rate = prof[kind] * N[:, None] + kappa_true * e_true * soup[None, :] + s_true * N[:, None] * chibar[None, :]
    counts = rng.poisson(rate).astype(np.float32)
    csr = DeviceCSR(sp.csr_matrix(counts), torch.device("cpu"))
    z = torch.tensor(np.eye(2)[kind] + 0.05 * rng.standard_normal((n, 2)), dtype=torch.float32)
    zero = torch.zeros(n, dtype=torch.long)
    depth = torch.tensor(counts.sum(1))
    kappa, swap, diag = exclusion_budget(
        csr, torch.arange(n), z, depth, zero, zero, torch.tensor([e_true]), torch.tensor(soup[None, :], dtype=torch.float32),
        torch.tensor(chibar[None, :], dtype=torch.float32),
    )
    assert kappa[0] is not None and swap[0] is not None, diag
    assert 0.7 * kappa_true < kappa[0] < 1.4 * kappa_true, (kappa, swap, diag)
    assert 0.6 * s_true < swap[0] < 1.5 * s_true, (kappa, swap, diag)
    # ambient only: the slope should come out near zero and kappa near the truth
    rate0 = prof[kind] * N[:, None] + kappa_true * e_true * soup[None, :]
    csr0 = DeviceCSR(sp.csr_matrix(rng.poisson(rate0).astype(np.float32)), torch.device("cpu"))
    k0, s0, d0 = exclusion_budget(
        csr0, torch.arange(n), z, depth, zero, zero, torch.tensor([e_true]), torch.tensor(soup[None, :], dtype=torch.float32),
        torch.tensor(chibar[None, :], dtype=torch.float32),
    )
    assert k0[0] is not None and 0.7 * kappa_true < k0[0] < 1.4 * kappa_true, (k0, s0, d0)
    assert s0[0] is None or s0[0] < 0.005, (k0, s0, d0)
    return kappa, swap, k0, s0


def test_synthetic_library():
    res = run()
    assert res["max_above_raw"] <= 0 and res["min_value"] >= 0 and res["integer"]
    assert res["cell_recall"] > 0.95, res
    assert res["false_cells"] < 30, res
    assert res["background_removed"] > 0.5, res
    assert res["native_retained"] > 0.98, res
    assert 0.6 < res["e_learned"] / res["e_true"] < 1.6, res


if __name__ == "__main__":
    test_apportion_bounds()
    test_gene_budget()
    test_exclusion_bound()
    print("exclusion budget (kappa, s; then ambient only):", test_exclusion_budget())
    print("apportion, gene budget and exclusion bounds ok")
    for est, lik in (("gene", "multinomial"), ("budget", "multinomial"), ("map", "multinomial"), ("gene", "dirmult")):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in run(est, lik).items()})
