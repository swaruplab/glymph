"""Spliced against unspliced molecules as a second negative control (splice.py, with multires.tree_silent)."""
import numpy as np
import scipy.sparse as sp
import torch

from glymph.multires import subclusters, tree_silent
from glymph.sparse import DeviceCSR
from glymph.splice import splice_profiles, splice_update, sums_of


def _nuclei(seed=0):
    """Two kinds of nuclei in a dirty library (a third of all counts is background, mostly B's RNA).
    B's genes sit in A at more than a fifth of their level in B, so the count rule cannot call them silent.
    Genes gW: the measured soup profile overstates them and A expresses them weakly, so their counts in A pass
    the count test although nearly all of them are A's own. Own RNA is half unspliced, background one twentieth."""
    rng = np.random.default_rng(seed)
    G = 600
    gA, gB, gW = np.arange(0, 100), np.arange(100, 200), np.arange(200, 230)
    profA, profB = np.ones(G), np.ones(G)
    profA[:230], profB[:230] = 0.0, 0.0
    profA[gA], profB[gB] = 3.0, 3.0
    profA[gW], profB[gW] = 0.35, 3.0
    profA, profB = profA / profA.sum(), profB / profB.sum()
    kind = np.r_[np.zeros(300, int), np.ones(300, int)]
    n = kind.size
    true_soup = 0.3 * profA + 0.7 * profB
    true_soup[gW] = 1e-4  # hardly any of gW reaches cells as background ...
    true_soup /= true_soup.sum()
    soup = true_soup.copy()
    soup[gW] = 1.5e-3     # ... but the measured profile says fifteen times more
    soup /= soup.sum()
    depth = rng.integers(2000, 3000, n)
    native = np.stack([rng.multinomial(depth[i], profA if kind[i] == 0 else profB) for i in range(n)])
    bg_level = 1300.0
    back = rng.poisson(bg_level * true_soup, (n, G))
    X = (native + back).astype(float)
    U = (rng.binomial(native, 0.5) + rng.binomial(back, 0.05)).astype(float)
    empty = rng.poisson(40 * soup, (3000, G))
    Ue = rng.binomial(empty, 0.05).sum(0).astype(float)
    return X, U, empty.sum(0).astype(float), Ue, soup, bg_level, kind, (gA, gB, gW)


def test_splicing_adds_and_vetoes_silent_genes():
    X, U, Xe, Ue, soup, bg, kind, (gA, gB, gW) = _nuclei()
    n, G = X.shape
    Xs = sp.csr_matrix(X.astype(np.float32))
    csr = DeviceCSR(Xs, torch.device("cpu"))
    pat = Xs.copy()
    pat.data[:] = 1
    Z = (pat + sp.csr_matrix(U).multiply(pat)).tocsr()
    Z.sort_indices()
    u = torch.from_numpy((Z.data - 1).astype(np.float32))
    lab = torch.from_numpy(kind)
    rows = torch.arange(n)
    z = torch.zeros(n, 2)
    z[rows, lab] = 4.0
    z += 0.3 * torch.randn(n, 2, generator=torch.Generator().manual_seed(0))
    fine, parent, _ = subclusters(z, lab, 2, min_cells=60, max_children=4, seed=0)
    depth = torch.from_numpy(X.sum(1))
    theta = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    Xc = csr.group_sums(rows, lab, 2)
    Uc = sums_of(csr, u, rows, lab, 2)
    Bc = torch.from_numpy(np.outer([300.0, 300.0], bg * soup))
    tested = torch.zeros(2, G, dtype=torch.bool)
    tested[1, gA] = True  # A's genes in B: silent by the counts
    big = torch.ones(2, dtype=torch.bool)
    prof = splice_profiles(torch.from_numpy(Xe), torch.from_numpy(Ue), Xc, Uc, Bc, tested, big, 1.0)
    inf = prof["informative"].numpy()
    assert inf[gA].mean() > 0.9 and inf[gB].mean() > 0.9, (inf[gA].mean(), inf[gB].mean())
    assert abs(float(prof["pi_nat"][gB].median()) - 0.5) < 0.08 and float(prof["pi_bg"][gB].median()) < 0.12, prof["diag"]
    args = (csr, rows, fine, parent, 2, depth, torch.zeros(n, dtype=torch.long), torch.tensor([bg]), torch.from_numpy(soup[None, :]),
            torch.zeros(G, dtype=torch.float64), theta, float(depth.median()), 1.0, 30.0, big, big)
    plain, _, _, _, _ = tree_silent(*args)
    counts, strip, _, _, diag = tree_silent(*args, splice={"u": u, "pi_bg": prof["pi_bg"], "pi_nat": prof["pi_nat"], "informative": prof["informative"]})
    fA = np.unique(fine.numpy()[kind == 0])
    fB = np.unique(fine.numpy()[kind == 1])
    plain, counts, strip = plain.numpy(), counts.numpy(), strip.numpy()
    # B's genes in A: too abundant for the count rule, but their molecules there are spliced like background
    assert plain[fA][:, gB].mean() < 0.1, plain[fA][:, gB].mean()
    assert strip[fA][:, gB].mean() > 0.9 and counts[fA][:, gB].mean() < 0.1, (strip[fA][:, gB].mean(), counts[fA][:, gB].mean())
    # gW in A: the counts pass (the profile overstates their background), the molecules are A's own: vetoed
    assert plain[fA][:, gW].mean() > 0.5, plain[fA][:, gW].mean()
    assert strip[fA][:, gW].mean() < 0.15, strip[fA][:, gW].mean()
    # nothing a cluster expresses is stripped
    assert strip[fA][:, gA].mean() < 0.02 and strip[fB][:, gB].mean() < 0.02 and strip[:, 230:].mean() < 0.02
    return diag


def test_update_moves_background_to_spliced_entries():
    share = torch.tensor([0.3, 0.3, 0.3, 0.3])
    v = torch.tensor([10.0, 10.0, 10.0, 10.0])
    u = torch.tensor([0.0, 5.0, 10.0, 5.0])
    pb, pn = torch.full((4,), 0.05), torch.full((4,), 0.5)
    inform = torch.tensor([True, True, True, False])
    new = splice_update(share, v, u, pb, pn, inform)
    assert float(new[0]) > 0.4 > 0.3 > float(new[1]) > float(new[2]) and float(new[2]) < 0.06 and float(new[3]) == float(share[3]), new
    # with the expected mix of molecules the share stays where it was
    exp_u = 10 * (0.3 * 0.05 + 0.7 * 0.5)
    same = splice_update(share[:1], v[:1], torch.tensor([exp_u]), pb[:1], pn[:1], inform[:1])
    assert abs(float(same) - 0.3) < 0.02, same


if __name__ == "__main__":
    d = test_splicing_adds_and_vetoes_silent_genes()
    test_update_moves_background_to_spliced_entries()
    print(f"splicing: {d['cluster_gene_pairs_silent_by_splicing_only']} cluster-gene pairs silent by splicing only, "
          f"{d['cluster_gene_pairs_vetoed_by_splicing']} vetoed; entry update: ok")
