"""The tier 2 metrics script on a synthetic library with known contamination per cell.

    python tests/test_kidney_metrics.py

A perfect tool (returns the native counts) must score a near-zero per-cell error, remove the
marker counts from non-PT cells and keep them in PT cells; a tool that does nothing must not.
"""

import os
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MARKERS = ["Slc34a1", "Miox", "Pck1", "Slc4a4", "Ttc36", "Lrp2", "Fbp1", "Cyp2e1", "Fut9", "Khk"]


def test_kidney_metrics():
    rng = np.random.default_rng(0)
    n, G = 600, 300
    genes = MARKERS + [f"Gene{i}" for i in range(G - len(MARKERS))]
    is_pt = rng.random(n) < 0.7
    prof_pt = rng.dirichlet(np.full(G, 0.5))
    prof_pt[: len(MARKERS)] += 0.02  # PT cells express the markers
    prof_pt /= prof_pt.sum()
    prof_other = rng.dirichlet(np.full(G, 0.5))
    prof_other[: len(MARKERS)] = 0.0  # the other cell type does not
    prof_other /= prof_other.sum()
    depth = rng.lognormal(np.log(4000), 0.4, n)
    frac = rng.uniform(0.02, 0.3, n)  # true background fraction per cell
    soup = 0.7 * prof_pt + 0.3 * prof_other
    native = rng.poisson(np.where(is_pt[:, None], prof_pt, prof_other) * (depth * (1 - frac))[:, None])
    bg = rng.poisson(soup[None, :] * (depth * frac)[:, None])
    raw = native + bg
    barcodes = np.array([f"BC{i:04d}-1" for i in range(n)])
    var = pd.DataFrame({"gene_id": [f"ENSMUSG{i:05d}" for i in range(G)], "gene_name": genes})
    var.index = var.gene_id

    def write(mat, path, bcs=barcodes):
        ad.AnnData(X=sp.csr_matrix(mat.astype(np.float32)), obs=pd.DataFrame(index=bcs), var=var.copy()).write_h5ad(path)

    tmp = tempfile.mkdtemp(prefix="glymph_kidney_")
    p = lambda f: os.path.join(tmp, f)  # noqa: E731
    write(raw, p("raw.h5ad"))
    write(native, p("perfect.h5ad"))
    write(raw, p("nothing.h5ad"))
    write(raw[: n // 2], p("half.h5ad"), barcodes[: n // 2])  # a tool that returns only half the cells
    truth = (raw.sum(1) - native.sum(1)) / np.maximum(raw.sum(1), 1)
    meta = pd.DataFrame({"replicate": "t1", "cell": barcodes, "celltype": np.where(is_pt, "PT", "DCT"),
                         "Strain": rng.choice(["BL6", "CAST", "SvImJ"], n), "contPerCell_binom": truth})
    meta.loc[5, "Strain"] = "doublet"  # must be dropped
    meta.to_csv(p("meta.tsv.gz"), sep="\t", index=False)
    open(p("markers.txt"), "w").write("\n".join(MARKERS) + "\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "code", "benchmark", "kidney_metrics.py"), "--name", "syn", "--replicate", "t1",
         "--kind", "cells", "--raw", p("raw.h5ad"), "--meta", p("meta.tsv.gz"), "--markers", p("markers.txt"),
         "--tool", f"perfect={p('perfect.h5ad')}", "--tool", f"nothing={p('nothing.h5ad')}", "--tool", f"half={p('half.h5ad')}",
         "--out", p("m.tsv"), "--per-cell", p("pc")],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    m = pd.read_csv(p("m.tsv"), sep="\t").set_index("tool")
    assert m.loc["raw", "n_eval_cells"] == n - 1 and m.loc["perfect", "n_missing"] == 0, m
    assert m.loc["half", "n_missing"] > 0.4 * n, m
    assert m.loc["perfect", "contam_rmsle"] < 1e-6 and m.loc["perfect", "contam_kendall"] > 0.99, m
    assert abs(m.loc["perfect", "removed_over_truth"] - 1) < 1e-6, m
    assert m.loc["perfect", "marker_removed_nonpt"] > 0.99 and m.loc["perfect", "marker_kept_pt"] > 0.6, m
    assert m.loc["perfect", "slc34a1_nonpt_after"] == 0 and m.loc["perfect", "slc34a1_nonpt_before"] > 5, m
    assert m.loc["nothing", "contam_rmsle"] > 0.05 and abs(m.loc["nothing", "marker_removed_nonpt"]) < 1e-9, m
    assert m.loc["perfect", "marker_log2fc"] > m.loc["raw", "marker_log2fc"] + 1, m
    assert (m[["n_above_raw", "n_negative", "n_noninteger"]].to_numpy() == 0).all(), m
    return m


if __name__ == "__main__":
    cols = ["n_eval_cells", "n_missing", "truth_pct", "removed_pct", "removed_over_truth", "contam_rmsle", "contam_kendall",
            "marker_removed_nonpt", "marker_kept_pt", "slc34a1_nonpt_before", "slc34a1_nonpt_after", "marker_log2fc"]
    print(test_kidney_metrics()[cols].round(4).T.to_string())
    print("kidney metrics ok")
