"""The block-wise paths used for large libraries give the same answer as the single-pass ones."""
import numpy as np
import scipy.sparse as sp
import torch

from glymph.sparse import DeviceCSR


def _random_csr(n=400, g=300, density=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n, g, density=density, format="csr", random_state=rng, data_rvs=lambda k: rng.integers(1, 9, k).astype(np.float32))
    return sp.csr_matrix(X)


def test_chunked_sums_match_dense():
    X = _random_csr()
    csr = DeviceCSR(X, torch.device("cpu"))
    rows = torch.arange(50, 350)
    lab = torch.from_numpy(np.random.default_rng(1).integers(0, 5, rows.numel()))
    dense = torch.from_numpy(X.toarray()).double()[rows]
    want = torch.zeros(5, X.shape[1], dtype=torch.float64).index_add_(0, lab, dense)
    for max_entries in (10**9, 5000, 300):  # one piece, several pieces, pieces of a few rows
        got = csr.group_sums(rows, lab, 5, max_entries=max_entries)
        assert torch.allclose(got, want)
    mask = torch.from_numpy(np.random.default_rng(2).random((5, X.shape[1])) < 0.3)
    want_y = (dense * mask[lab].double()).sum(1)
    for max_entries in (10**9, 5000, 300):
        assert torch.allclose(csr.masked_row_sums(rows, lab, mask, max_entries=max_entries), want_y)
    pieces = csr.row_chunks(rows, 300)
    assert pieces[0][0] == 0 and pieces[-1][1] == rows.numel() and all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))


def test_take_returns_long_columns():
    csr = DeviceCSR(_random_csr(), torch.device("cpu"))
    r, c, v, pos = csr.take(torch.arange(10))
    assert c.dtype == torch.int64 and csr.indices.dtype == torch.int32
    x, _ = csr.dense(torch.arange(10))
    assert float(x.sum()) == float(v.sum())


def test_large_output_path_matches_single_pass():
    """denoise() with the entry limit forced to zero must equal the gene-only rule in one pass."""
    from glymph import api

    X = _random_csr(n=300, g=200, density=0.3, seed=3)
    csr = DeviceCSR(X, torch.device("cpu"))
    n = X.shape[0]
    rng = np.random.default_rng(4)
    share = torch.from_numpy(rng.random(csr.data.numel()).astype(np.float32) * 0.3)

    class Post:
        pass

    class Fake:
        cell_log_kappa = None

        def lk(self, b):
            return None

        def sc(self, b):
            return None

        def posterior(self, x, nz, unit, pool, prior_logit, lk, sc):
            r, c, v, pos = nz
            p = Post()
            p.share_amb, p.share_swp = share[pos], torch.zeros_like(v)
            p.cell_prob = torch.ones(x.shape[0])
            p.eps = torch.ones(x.shape[0])
            p.z = torch.zeros(x.shape[0], 2)
            p.N = x.sum(1)
            return p

    cfg = api.Config()
    cfg.keep_cell_totals = False
    zeros = torch.zeros(n, dtype=torch.long)
    thr = float(np.median(np.asarray(X.sum(1)).ravel()))
    ref = api.denoise(Fake(), csr, zeros, zeros, torch.zeros(n), cfg, cell_threshold=thr)
    cfg.cell_totals_max_entries = 0  # forces the block-wise path
    big = api.denoise(Fake(), csr, zeros, zeros, torch.zeros(n), cfg, cell_threshold=thr)
    assert np.array_equal(ref["data"], big["data"])
    assert np.allclose(ref["removed"], big["removed"])
    assert (big["data"] <= csr.data.numpy()).all() and (big["data"] >= 0).all()



def test_allowance_capped_by_background():
    """With a cap, a gene gives up at most (1 + cap) times its expected background, and nothing where it expects none."""
    from glymph.thinning import gene_budget, gene_cell_budget

    rng = np.random.default_rng(7)
    n, G = 6000, 40
    g = torch.from_numpy(rng.integers(0, G, n))
    r = torch.from_numpy(rng.integers(0, 300, n))
    v = torch.from_numpy(rng.integers(1, 30, n).astype(np.float32))
    p = torch.from_numpy((rng.random(n) * 0.02).astype(np.float32))
    p[g < 10] = 1e-6  # ten genes with no expected background
    bg = torch.zeros(G, dtype=torch.float64).index_add_(0, g, (v * p.clamp(1e-6, 1 - 1e-6)).double())
    for fn in (lambda cap: gene_budget(g, v, p, G, fpr=0.01, cap=cap), lambda cap: gene_cell_budget(g, r, v, p, G, 300, fpr=0.01, cap=cap)):
        free, capped = fn(None), fn(0.2)
        rm_free = torch.zeros(G, dtype=torch.float64).index_add_(0, g, free.double())
        rm_cap = torch.zeros(G, dtype=torch.float64).index_add_(0, g, capped.double())
        assert float(rm_cap[:10].sum()) == 0.0 and float(rm_free[:10].sum()) > 0.0
        assert bool((rm_cap <= torch.round(1.2 * bg) + 1).all()) and bool((rm_cap <= rm_free + 1).all())
        assert bool((capped <= v).all()) and bool((capped >= 0).all()) and bool((capped == torch.round(capped)).all())


if __name__ == "__main__":
    test_chunked_sums_match_dense()
    test_take_returns_long_columns()
    test_large_output_path_matches_single_pass()
    test_allowance_capped_by_background()
    print("large-library paths and capped allowance: ok")
