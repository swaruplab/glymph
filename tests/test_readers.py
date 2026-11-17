#!/usr/bin/env python
"""Vendor readers on tiny synthetic files: BD Rhapsody dense and long tables, and a Parse
split-pipe folder with GEO-style prefixed file names."""
import gzip
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from glymph.io import read_counts, read_mtx_dir  # noqa: E402


def main():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    M = rng.poisson(0.3, (7, 5))
    M[3] = 0
    genes = ["A", "B", "CD3|CD3E|AHS1|pAbO", "C", "D"]
    keep = [0, 1, 3, 4]
    rows = M[:, keep].sum(1) > 0

    with gzip.open(f"{d}/x_RSEC_MolsPerCell_Unfiltered.csv.gz", "wt") as f:
        f.write("####\n## BD\n####\nCell_Index," + ",".join(genes) + "\n")
        for i, r in enumerate(M):
            f.write(f"{100 + i}," + ",".join(map(str, r)) + "\n")
    a = read_counts(f"{d}/x_RSEC_MolsPerCell_Unfiltered.csv.gz")
    assert a.shape == (rows.sum(), 4) and (a.X.toarray() == M[rows][:, keep]).all(), "BD dense table"
    assert list(a.obs_names) == [str(100 + i) for i in np.flatnonzero(rows)]

    with gzip.open(f"{d}/x_Expression_Data_Unfiltered.st.txt.gz", "wt") as f:
        f.write("####\n## BD\n####\nCell_Index\tGene\tRSEC_Reads\tRaw_Molecules\tRSEC_Adjusted_Molecules\n")
        for i, r in enumerate(M):
            for j, v in enumerate(r):
                if v:
                    f.write(f"{100 + i}\t{genes[j]}\t{v * 3}\t{v + 1}\t{v}\n")
    b = read_counts(f"{d}/x_Expression_Data_Unfiltered.st.txt.gz")
    order = [list(b.var.gene_id).index(g) for g in ["A", "B", "C", "D"]]
    assert (b.X.toarray()[:, order] == M[rows][:, keep]).all(), "BD long table"

    os.makedirs(f"{d}/parse")
    P = sp.csr_matrix(rng.poisson(0.5, (6, 4)))
    mmwrite(f"{d}/parse/GSE1_unfiltered_DGE.mtx", P)
    pd.DataFrame({"gene_id": list("wxyz"), "gene_name": list("WXYZ"), "genome": "mm10"}).to_csv(
        f"{d}/parse/GSE1_unfiltered_all_genes.csv", index=False)
    pd.DataFrame({"bc_wells": [f"01_02_{i:02d}" for i in range(6)], "bc1_well": ["A1"] * 3 + ["A2"] * 3}).to_csv(
        f"{d}/parse/GSE1_unfiltered_cell.csv", index=False)
    c = read_mtx_dir(f"{d}/parse")
    nz = np.asarray(P.sum(1)).ravel() > 0
    assert (c.X.toarray() == P.toarray()[nz]).all() and "bc1_well" in c.obs, "Parse folder"
    print("readers ok:", a.shape, b.shape, c.shape)


if __name__ == "__main__":
    main()
