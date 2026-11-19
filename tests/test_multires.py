"""Multi-resolution silent sets and the reproducible embedding (multires.py)."""
import numpy as np
import scipy.sparse as sp
import torch

from glymph.multires import pca_embedding, subclusters, tree_silent
from glymph.sparse import DeviceCSR


def _tissue(seed=0):
    """Two clusters, A and B. A holds a sub-population (60 of 300 cells) with genes of its own.
    One cell of B also carries A's genes (a doublet). Background: 40 counts per cell, soup profile."""
    rng = np.random.default_rng(seed)
    G = 600
    gA, gB, gS = np.arange(0, 100), np.arange(100, 200), np.arange(200, 250)
    soup = rng.gamma(2.0, 1.0, G)
    soup /= soup.sum()
    kind = np.r_[np.zeros(240, int), np.full(60, 2), np.ones(300, int)]  # 0: A, 2: A's sub-population, 1: B
    n = kind.size
    prof = np.full((3, G), 1.0)
    prof[:, :250] = 0.0
    prof[0, gA] = 3.0
    prof[2, gA] = 3.0
    prof[2, gS] = 6.0
    prof[1, gB] = 3.0
    prof /= prof.sum(1, keepdims=True)
    depth = rng.integers(1500, 3000, n)
    X = np.stack([rng.multinomial(depth[i], prof[kind[i]]) for i in range(n)]).astype(float)
    X[400] += rng.multinomial(800, prof[0])  # the doublet: a B cell with a cell of A in it
    bg = 40.0
    X += rng.poisson(bg * soup, (n, G))
    lab = (kind == 1).astype(int)  # coarse clusters: A with its sub-population, and B
    z = np.zeros((n, 3))
    z[np.arange(n), kind] = 4.0
    z += rng.normal(0, 0.3, z.shape)
    return X, soup, bg, lab, kind, z, (gA, gB, gS)


def test_tree_silent():
    X, soup, bg, lab, kind, z, (gA, gB, gS) = _tissue()
    n, G = X.shape
    csr = DeviceCSR(sp.csr_matrix(X.astype(np.float32)), torch.device("cpu"))
    lab_t = torch.from_numpy(lab)
    fine, parent, centres = subclusters(torch.from_numpy(z).float(), lab_t, 2, min_cells=40, max_children=4, seed=0)
    assert int(parent.numel()) >= 3 and bool((parent[fine] == lab_t).all()), "sub-clusters are not nested in their clusters"
    depth = torch.from_numpy(X.sum(1))
    theta = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    silent, strip, rate, B, diag = tree_silent(
        csr, torch.arange(n), fine, parent, 2, depth, torch.zeros(n, dtype=torch.long), torch.tensor([bg]),
        torch.from_numpy(soup[None, :]), torch.zeros(G, dtype=torch.float64), theta, float(depth.median()), 1.0, 30.0,
        torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    assert torch.equal(silent, strip), "without splicing the two sets are the same"
    f_of = fine.numpy()
    sub_f = np.bincount(f_of[kind == 2]).argmax()          # the fine cluster of A's sub-population
    a_f = [f for f in np.unique(f_of[kind == 0]) if f != sub_f]
    b_f = np.unique(f_of[kind == 1])
    s = silent.numpy()
    assert (kind[f_of == sub_f] == 2).mean() > 0.9, "the sub-population did not get its own sub-cluster"
    # B does not express A's genes nor the sub-population's, although one doublet in B carries them
    for f in b_f:
        assert s[f, gA].mean() > 0.9 and s[f, gS].mean() > 0.9, (f, s[f, gA].mean(), s[f, gS].mean())
        assert s[f, gB].mean() < 0.05
    # the rest of A: B's genes are silent, and so are the sub-population's (set aside, the rest is at background)
    for f in a_f:
        assert s[f, gB].mean() > 0.9 and s[f, gS].mean() > 0.9 and s[f, gA].mean() < 0.05, (f, s[f, gB].mean(), s[f, gS].mean())
    # the sub-population keeps its own genes: the sub-cluster overrules its parent
    assert s[sub_f, gS].mean() < 0.05 and s[sub_f, gB].mean() > 0.9, (s[sub_f, gS].mean(), s[sub_f, gB].mean())
    assert s[:, 250:].mean() < 0.02, "genes every cell expresses were called silent"
    assert diag["isolated_carrier_entries"] > 0
    return int(parent.numel()), diag


def test_pca_embedding_is_reproducible_and_separates():
    X, soup, bg, lab, kind, z, _ = _tissue(1)
    csr = DeviceCSR(sp.csr_matrix(X.astype(np.float32)), torch.device("cpu"))
    rows = torch.arange(X.shape[0])
    a = pca_embedding(csr, rows, rows, n_genes=300, n_comp=5, seed=0)
    b = pca_embedding(csr, rows, rows, n_genes=300, n_comp=5, seed=0)
    assert torch.equal(a, b), "two calls with the same seed differ"
    c = pca_embedding(csr, rows[::2], rows, n_genes=300, n_comp=5, seed=0)
    assert c.shape == a.shape
    # the three kinds of cells are apart on the first components
    cen = torch.stack([a[torch.from_numpy(kind == j)].mean(0) for j in range(3)])
    spread = max(float(a[torch.from_numpy(kind == j)].std(0).max()) for j in range(3))
    assert float(torch.cdist(cen, cen)[0, 1]) > 5 * spread and float(torch.cdist(cen, cen)[0, 2]) > 3 * spread


def test_shared_level():
    """A gene at the same multiple of the fit in every cluster that does not express it is silent there, at that multiple."""
    from glymph.multires import shared_level

    g = torch.Generator().manual_seed(0)
    k, per = 8, 200
    kinds = ["fit", "double everywhere", "double, two clusters express", "five times everywhere", "rare double", "one cluster only"]
    G = per * len(kinds)
    B = torch.rand(k, G, generator=g, dtype=torch.float64) * 150 + 50  # fitted background, 50 to 200 counts per pair
    mult = torch.ones(k, G, dtype=torch.float64)
    cand = torch.ones(k, G, dtype=torch.bool)
    sl = {name: slice(i * per, (i + 1) * per) for i, name in enumerate(kinds)}
    mult[:, sl["double everywhere"]] = 2.0
    mult[:, sl["double, two clusters express"]] = 2.0
    mult[:2, sl["double, two clusters express"]] = 12.0
    mult[:, sl["five times everywhere"]] = 5.0
    B[:, sl["rare double"]] = 1.0 + torch.rand(k, per, generator=g, dtype=torch.float64) * 2.0
    mult[:, sl["rare double"]] = 2.0
    mult[:, sl["one cluster only"]] = 2.0
    cand[1:, sl["one cluster only"]] = False  # a candidate in a single cluster: nothing to compare with
    X = torch.poisson(B * mult, generator=g)
    member, factor = shared_level(X, B, cand, need=30.0)
    frac = lambda name, rows=slice(None): float(member[rows, sl[name]].double().mean())  # noqa: E731
    assert abs(float(factor[sl["fit"]].median()) - 1.0) < 0.02 and float((factor[sl["fit"]] - 1.0).abs().max()) < 0.2, "genes at the fit stay at the fit"
    mult2 = mult.clone(); mult2[:, sl["fit"]] = 0.6
    _, low = shared_level(torch.poisson(B * mult2, generator=g), B, cand, need=30.0)
    assert abs(float(low[sl["fit"]].median()) - 0.6) < 0.05, "a gene below the fit in every cluster gets its lower level"
    assert frac("double everywhere") > 0.97, frac("double everywhere")
    assert abs(float(factor[sl["double everywhere"]].median()) - 2.0) < 0.1
    assert frac("double, two clusters express", slice(2, None)) > 0.97 and frac("double, two clusters express", slice(0, 2)) == 0.0
    assert frac("five times everywhere") == 0.0, "beyond the largest multiple nothing is called silent"
    assert frac("one cluster only") == 0.0, "one cluster cannot show a shared level"
    assert 0.3 < frac("rare double") <= 1.0, frac("rare double")  # rare genes join where their few counts allow
    return {name: round(frac(name), 3) for name in kinds}


if __name__ == "__main__":
    print("level shared across clusters (share of pairs called silent):", test_shared_level())
    f, diag = test_tree_silent()
    test_pca_embedding_is_reproducible_and_separates()
    print(f"multi-resolution silent sets: {f} sub-clusters, {diag['genes_silent_after_setting_subclusters_aside']} genes silent after setting "
          f"sub-clusters aside, {diag['subcluster_gene_pairs_overruling_parent']} overruled, {diag['isolated_carrier_entries']} isolated carrier entries; "
          "embedding reproducible: ok")
