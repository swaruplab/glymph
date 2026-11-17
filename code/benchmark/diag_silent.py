#!/usr/bin/env python
"""Rerun the silent-gene calibration on a finished Glymph output, and compare with the truth.

Uses the output's latent space, raw counts and soup, so no model is trained. For kidney libraries
the truth is the authors' per-cell genotype-based background (fraction x raw depth), summarised as
a line in depth; the estimate should give the same counts per cell.

    python diag_silent.py <out.h5ad> <report.json> [--kidney rep2 --meta cell_metadata.tsv.gz] [--device cuda:0]
"""
import argparse
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.exclusion import silent_budget  # noqa: E402
from glymph.sparse import DeviceCSR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("h5ad")
ap.add_argument("report")
ap.add_argument("--kidney")
ap.add_argument("--meta")
ap.add_argument("--device", default="cpu")
ap.add_argument("--delta", type=float, default=0.2)
ap.add_argument("--clusters", type=int, default=20)
a = ap.parse_args()

rep = json.load(open(a.report))
out = ad.read_h5ad(a.h5ad)
X = sp.csr_matrix(out.layers["raw"]).astype(np.float32)
N = np.asarray(X.sum(1)).ravel()
conf = float(rep["priors"].get("confident_threshold", rep["priors"]["cell_threshold"]))
rows = np.flatnonzero(out.obs["glymph_is_cell"].to_numpy() & (N >= conf))
e = float(list(rep["e_unit"].values())[0])
soup = out.var["glymph_soup_mean"].to_numpy(float)
keep = np.flatnonzero((np.asarray(X.sum(0)).ravel() > 0) & (soup > 0))
X = X[:, keep].tocsr()
soup = soup[keep] / soup[keep].sum()
chibar = np.asarray(X[rows].sum(0)).ravel().astype(float)
chibar = (chibar + 1e-9) / (chibar + 1e-9).sum()
dev = torch.device(a.device)
csr = DeviceCSR(X, dev)
t = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=dev)  # noqa: E731
zero = torch.zeros(rows.size, dtype=torch.long, device=dev)
learned = float(rep["swap_calibration"]["swap_learned"][0])
k_hat, s_hat, diag, _ = silent_budget(csr, t(rows, torch.long), t(out.obsm["X_glymph"][rows]), t(N[rows], torch.float64), zero, zero,
                                   t([e]), t(soup)[None, :], t(chibar)[None, :], t([learned]), n_clusters=a.clusters, delta=a.delta)
d = diag["0"]
name = os.path.basename(a.h5ad)[:-5]
Nm = float(np.median(N[rows]))
print(f"{name}: confident cells {rows.size}, median depth {Nm:.0f}, e {e:.0f}, learned s {learned:.4f}")
kd, nref = d.get("kappa_depth", 0.0), d.get("depth_ref", Nm)
bg = lambda n: (k_hat[0] + kd * n / nref) * e + s_hat[0] * n  # noqa: E731
print(f"   estimate: kappa {k_hat[0]}, kappa_depth {kd}, s {s_hat[0]}  ->  background per cell at median depth {'' if k_hat[0] is None else round(bg(Nm))}"
      f"   (ridge {d.get('ridge')}, cells fitted {d.get('cells_fitted')}, trimmed {d.get('cells_trimmed')}, clusters dropped {d.get('clusters_dropped')}, {d.get('reason', '')})")
h = d.get("ridge_histogram")
if h:
    print("   histogram of log10(observed / expected background), bins of 0.1 from -2 to 2 (showing -1.0 to 1.0):")
    print("   ", " ".join(f"{v}" for v in h[10:30]))
for row in d.get("clusters", []):
    print(f"      cluster {row['cluster']:2d} cells {row['cells']:5d} candidates {row['candidates']:5d} silent {row['genes']:5d} ratio {row.get('ratio')} used {row.get('used')}")
if a.kidney:
    m = pd.read_csv(a.meta, sep="\t")
    m = m[(m.replicate == a.kidney) & m.contPerCell_binom.notna()].drop_duplicates("cell").set_index("cell")
    idx = out.obs_names.get_indexer(m.index)
    ok = idx >= 0
    Nt = N[idx[ok]]
    b = m.contPerCell_binom.to_numpy(float)[ok] * Nt
    q = np.quantile(Nt, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(Nt, q[1:-1]), 0, 9)
    mN = np.array([np.median(Nt[bins == i]) for i in range(10)])
    mb = np.array([np.median(b[bins == i]) for i in range(10)])
    mean_b = np.array([b[bins == i].mean() for i in range(10)])
    (c0, c1), *_ = np.linalg.lstsq(np.vstack([np.ones(10), mN]).T, mb, rcond=None)
    print(f"   truth ({ok.sum()} cells): median background per cell {np.median(b):.0f}, mean {b.mean():.0f}; by depth decile, median: "
          + " ".join(f"{v:.0f}" for v in mb) + f"; line {c0:.0f} + {c1:.4f} N -> kappa {c0 / e:.2f}, s {c1:.4f}")
    if k_hat[0] is not None:
        est = bg(mN)
        print("   estimate at the same depths: " + " ".join(f"{v:.0f}" for v in est) + f";  estimate / truth median {np.median(est / mb):.2f}, / truth mean {np.median(est / mean_b):.2f}")
