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


if __name__ == "__main__":
    test_chunked_sums_match_dense()
    test_take_returns_long_columns()
    test_large_output_path_matches_single_pass()
    print("large-library paths: ok")
