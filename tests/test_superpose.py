#!/usr/bin/env python
"""A cluster that is another cluster plus background-like RNA takes that cluster's silent genes; a cell type keeps its own.

Four cell types in a clean library (background about 1.3% of a cell). 150 cells of the first type carry 10% extra
RNA with the average cell profile and sit apart in the embedding, as heavily contaminated cells do. Without the rule their cluster "expresses" every other type's genes
and nothing is stripped from them; with it they get the first type's silent genes. The rarest true cell type
(a tenth of the cells) must keep its own set either way.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from glymph.exclusion import silent_budget  # noqa: E402
from glymph.sparse import DeviceCSR  # noqa: E402


def library(seed=0, n=6000, n_dirty=150, extra=0.10, kappa=0.3, s=0.003, e=100.0):
    rng = np.random.default_rng(seed)
    G, K, spec = 2000, 4, 300
    prof = np.zeros((K, G))
    shared = rng.gamma(0.5, 1.0, G - K * spec)
    for k in range(K):
        prof[k, K * spec:] = 0.6 * shared / shared.sum() * rng.lognormal(0, 0.2, shared.size)
        own = rng.gamma(0.5, 1.0, spec)
        prof[k, k * spec:(k + 1) * spec] = 0.4 * own / own.sum()
        prof[k] /= prof[k].sum()
    typ = rng.choice(K, n, p=[0.5, 0.25, 0.15, 0.1])
    dirty = np.zeros(n, bool)
    dirty[rng.choice(np.flatnonzero(typ == 0), n_dirty, replace=False)] = True
    depth = rng.lognormal(np.log(3000), 0.4, n)
    chibar = (prof[typ] * depth[:, None]).sum(0)
    chibar /= chibar.sum()
    soup = chibar * rng.lognormal(0, 0.3, G)
    soup /= soup.sum()
    X = rng.poisson(prof[typ] * depth[:, None]).astype(np.float64)
    native = X.sum(1)
    X += rng.poisson(kappa * e * rng.gamma(10.0, 0.1, n)[:, None] * soup[None, :])
    X += rng.poisson(s * native[:, None] * chibar[None, :])
    X[dirty] += rng.poisson(extra * native[dirty][:, None] * chibar[None, :])
    z = np.concatenate([np.eye(K)[typ], dirty[:, None].astype(float)], 1) + rng.normal(0, 0.08, (n, K + 1))
    return sp.csr_matrix(X), z, soup, chibar, typ, dirty, spec


def run(superpose):
    X, z, soup, chibar, typ, dirty, spec = library()
    csr = DeviceCSR(X.astype(np.float32), torch.device("cpu"))
    n = X.shape[0]
    zero = torch.zeros(n, dtype=torch.long)
    _, _, diag, states = silent_budget(
        csr, torch.arange(n), torch.as_tensor(z, dtype=torch.float32), torch.as_tensor(np.asarray(X.sum(1)).ravel()),
        zero, zero, torch.tensor([100.0]), torch.as_tensor(soup, dtype=torch.float32)[None, :],
        torch.as_tensor(chibar, dtype=torch.float32)[None, :], torch.tensor([0.01]), n_clusters=8, superpose=superpose)
    st = states[0]
    assert st is not None, diag
    cl = torch.cdist(torch.as_tensor(z, dtype=torch.float64), st["centres"]).argmin(1).numpy()
    mask = st["mask_broad"].numpy()
    other = np.zeros(mask.shape[1], bool)
    other[spec:4 * spec] = True  # the genes of the three other cell types
    silent_dirty = float(mask[cl[dirty]][:, other].mean())  # share of those genes that are silent for the dirty cells
    rare = typ == 3
    own = np.zeros(mask.shape[1], bool)
    own[3 * spec:4 * spec] = True
    silent_own_rare = float(mask[cl[rare]][:, own].mean())  # the rare cell type's own genes must not be silent for it
    return silent_dirty, silent_own_rare, diag["0"]


if __name__ == "__main__":
    off = run(False)
    on = run(True)
    merged = on[2].get("clusters_given_a_larger_clusters_silent_genes", [])
    print(f"other cell types' genes silent for the 150 heavily contaminated cells: {off[0]:.2f} without the rule, {on[0]:.2f} with it; "
          f"the rare cell type's own genes silent for it: {off[1]:.3f} and {on[1]:.3f}; clusters given a larger cluster's set: {merged}")
    assert off[0] < 0.5, "without the rule the contaminated cluster keeps most of the other types' genes"
    assert on[0] > 0.9, "with the rule the contaminated cells get the silent genes of their own type"
    assert on[1] < 0.02 and len(merged) >= 1, "a cell type keeps its own silent set"
    assert all(r["share_of_counts"] <= 0.2 and r["match"] >= 0.5 for r in merged)
    print("background-like clusters: ok")
