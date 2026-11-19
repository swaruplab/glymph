"""The profile test that tells a heavily contaminated cell from one with a second expression program."""
import numpy as np
import scipy.sparse as sp
import torch

from glymph.exclusion import background_match
from glymph.sparse import DeviceCSR


def test_background_match_separates_profiles():
    rng = np.random.default_rng(0)
    G, n = 3000, 60
    soup = rng.gamma(0.3, 1.0, G)
    soup /= soup.sum()
    other = np.zeros(G)  # a second cell's expression: a few hundred genes, unrelated to the soup
    idx = rng.choice(G, 300, replace=False)
    other[idx] = rng.gamma(0.5, 1.0, idx.size)
    mask = np.zeros((2, G), bool)  # two clusters with different silent genes
    mask[0, :2000] = True
    mask[1, 1000:] = True
    lab = rng.integers(0, 2, n)
    X = np.zeros((n, G))
    for i in range(n):
        m = mask[lab[i]]
        prof = (soup if i < n // 2 else other) * m
        X[i] = rng.multinomial(int(rng.integers(80, 600)), prof / prof.sum())
        X[i, ~m] = rng.poisson(2.0, int((~m).sum()))  # the cell's own genes must not matter
    csr = DeviceCSR(sp.csr_matrix(X.astype(np.float32)), torch.device("cpu"))
    args = (csr, torch.arange(n), torch.from_numpy(lab), torch.from_numpy(mask), torch.ones(n, dtype=torch.float64),
            torch.from_numpy(soup[None, :]).float(), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.float64), torch.from_numpy(soup).float())
    # cluster profiles: what each cluster expresses outside its silent genes; cells 0..n/2 follow their cluster
    rate = rng.gamma(0.3, 1.0, (2, G)) * ~mask
    stranger = rng.gamma(0.3, 1.0, G) * (rng.random(G) < 0.3)  # a cell of another kind placed in the cluster: mostly other genes
    for i in range(n):
        m = mask[lab[i]]
        prof = (rate[lab[i]] if i < n // 2 else stranger) * ~m
        X[i, ~m] = rng.multinomial(int(rng.integers(500, 3000)), prof / prof.sum())[~m]
    csr = DeviceCSR(sp.csr_matrix(X.astype(np.float32)), torch.device("cpu"))
    args = (csr,) + args[1:]
    got, member = background_match(*args, rate=torch.from_numpy(rate))
    again = background_match(*args, rate=torch.from_numpy(rate), chunk=7)
    assert torch.allclose(got, again[0]) and torch.allclose(member, again[1]), "the result depends on the chunk size"
    own_share = torch.from_numpy(np.array([X[i, ~mask[lab[i]]].sum() / X[i].sum() for i in range(n)]))
    assert bool(((member[: n // 2] / own_share[: n // 2]) > 0.85).all()), member[: n // 2] / own_share[: n // 2]
    assert float(member[n // 2:].max()) < 0.3, member[n // 2:]
    bg, second = got[: n // 2], got[n // 2:]
    assert 0.85 < float(bg.mean()) < 1.15 and float(bg.min()) > 0.6, bg
    assert float(second.max()) < 0.5, second
    # a row with no counts on its silent genes has no profile to compare
    X[0, mask[lab[0]]] = 0
    csr0 = DeviceCSR(sp.csr_matrix(X.astype(np.float32)), torch.device("cpu"))
    assert bool(torch.isnan(background_match(csr0, *args[1:])[0][0]))
    return float(bg.mean()), float(bg.min()), float(second.max()), float((member[: n // 2] / own_share[: n // 2]).min()), float(member[n // 2:].max())


if __name__ == "__main__":
    print("background match: background-like cells mean %.2f (lowest %.2f), second-program cells at most %.2f; "
          "cluster match over own share: members at least %.2f, strangers at most %.2f: ok" % test_background_match_separates_profiles())
