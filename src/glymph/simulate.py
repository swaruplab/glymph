"""Small synthetic libraries with known native and background counts, for tests only.

The simulator follows the same story as the model (constant ambient amount per
partition plus a depth-proportional swapped fraction), so passing on it shows the
code is wired correctly. It says nothing about accuracy on real data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def simulate(
    n_cells: int = 1500,
    n_empty: int = 6000,
    n_tail: int = 8000,
    n_genes: int = 1500,
    n_types: int = 4,
    cell_size: float = 4000.0,
    ambient_level: float = 80.0,
    swap_fraction: float = 0.01,
    eps_alpha: float = 10.0,
    species_mix: bool = False,
    seed: int = 0,
):
    """Returns (raw AnnData, truth dict with the native count matrix and labels).

    species_mix=True makes a two-species "barnyard": two cell types with disjoint gene
    sets, gene ids prefixed GRCh38_ and mm10_, as in the 10x human + mouse mixtures.
    """
    rng = np.random.default_rng(seed)
    if species_mix:
        n_types = 2
    block = n_genes // (n_types + 1)
    profiles = np.zeros((n_types, n_genes))
    for k in range(n_types):
        base = rng.gamma(0.3, 1.0, n_genes)
        base[: n_types * block] *= 1e-4  # other types' marker blocks are essentially off
        base[k * block : (k + 1) * block] = rng.gamma(0.5, 6.0, block)
        if species_mix:
            half = n_genes // 2
            base[half:] = 0.0 if k == 0 else base[half:]
            base[:half] = 0.0 if k == 1 else base[:half]
        profiles[k] = base / base.sum()

    ctype = rng.integers(0, n_types, n_cells)
    size_mult = np.array([1.0, 0.5, 2.0, 0.25] * (n_types // 4 + 1))[:n_types]
    d = rng.lognormal(np.log(cell_size * size_mult[ctype]), 0.35).astype(int) + 50
    native = np.zeros((n_cells, n_genes), dtype=np.int64)
    for i in range(n_cells):
        p = profiles[ctype[i]] * rng.gamma(20.0, 1 / 20.0, n_genes)
        native[i] = rng.multinomial(d[i], p / p.sum())

    chibar = native.sum(0) / native.sum()
    soup = 0.7 * chibar + 0.3 * profiles[0]  # soup is not just the average cell
    soup /= soup.sum()

    eps_c = rng.gamma(eps_alpha, 1 / eps_alpha, n_cells)
    amb = rng.poisson(eps_c[:, None] * ambient_level * soup[None, :])
    swp = rng.poisson(swap_fraction / (1 - swap_fraction) * d[:, None] * chibar[None, :])
    cells = native + amb + swp

    eps_e = rng.gamma(eps_alpha, 1 / eps_alpha, n_empty)
    empty = rng.poisson(eps_e[:, None] * ambient_level * soup[None, :])
    tail = rng.poisson(rng.uniform(1, 8, n_tail)[:, None] * soup[None, :])

    X = sp.csr_matrix(np.vstack([cells, empty, tail]).astype(np.int32))
    kind = np.array(["cell"] * n_cells + ["empty"] * n_empty + ["tail"] * n_tail)
    names = np.array([f"bc{i:07d}" for i in range(X.shape[0])])
    perm = rng.permutation(X.shape[0])
    prefix = [("GRCh38_" if j < n_genes // 2 else "mm10_") if species_mix else "" for j in range(n_genes)]
    var = pd.DataFrame({"gene_id": [f"{prefix[j]}g{j:05d}" for j in range(n_genes)]})
    var["gene_name"] = var["gene_id"]
    if species_mix:
        var["genome"] = [p.rstrip("_") for p in prefix]
    var.index = var["gene_id"].to_numpy()
    raw = ad.AnnData(X=X[perm], obs=pd.DataFrame({"kind": kind[perm]}, index=names[perm]), var=var)
    keep = np.diff(raw.X.indptr) > 0
    raw = raw[keep].copy()
    truth = {
        "native": pd.DataFrame(native, index=names[:n_cells], columns=var.index),
        "background": pd.DataFrame(amb + swp, index=names[:n_cells], columns=var.index),
        "cell_type": pd.Series(ctype, index=names[:n_cells]),
        "ambient_level": ambient_level,
        "swap_fraction": swap_fraction,
    }
    return raw, truth
