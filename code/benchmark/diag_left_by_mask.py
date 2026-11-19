#!/usr/bin/env python
"""Cross-species counts that Glymph leaves, split by what its silent test said (species mixtures; development).

Uses the dump of the silent test (GLYMPH_DEBUG_DUMP) and the output of the same run. For the scored cells that took
part in the calibration, every cross-species entry is put in the class of its gene in the cell's sub-cluster:

  silent            stripped by the exclusion constraint
  overruled         silent for the cluster, kept for this sub-cluster (carriers)
  not a candidate   the cluster's rate is above a fifth of the top cluster's
  above the level   tested and above the level
  rare, above       too rare to test one by one, above what the level allows
  cluster unused    small or dropped cluster

and by the cell: left alone as doublet-like or not, and how dirty it is.

    python diag_left_by_mask.py --raw data/std/LIB.h5ad --dump out_dbg/TAG/LIB.pool0.pt --out out_dbg/TAG/LIB.h5ad \
        --per-cell out/metrics/per_cell/LIB.TOOL.tsv.gz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_mix_metrics import gene_species, read_counts  # noqa: E402

ap = argparse.ArgumentParser()
for k_ in ("raw", "dump", "out", "per-cell"):
    ap.add_argument("--" + k_, required=True)
a = ap.parse_args()
raw = read_counts(a.raw)
idx = np.load(a.dump[: a.dump.rindex(".pool")] + ".index.npz")
gi, analyzed = idx["genes"], idx["analyzed"]
d = torch.load(a.dump, map_location="cpu", weights_only=False)
spg = gene_species(raw.var)[gi]
gname = (raw.var["gene_name"].astype(str).to_numpy() if "gene_name" in raw.var else raw.var_names.astype(str).to_numpy())[gi]
pc = pd.read_csv(a.per_cell, sep="\t").set_index("barcode")
rows = d["rows"].numpy()  # rows of the analysed matrix, one per calibration cell
names = raw.obs_names[analyzed[rows]]
keepc = names.isin(pc.index)
print(f"{os.path.basename(a.dump)}: {len(names)} calibration cells, {int(keepc.sum())} of them scored ({len(pc)} scored in all)")
X = sp.csr_matrix(raw.X)[analyzed[rows]][:, gi].astype(np.float64)
o = read_counts(a.out, integer=False)
og = pd.Index(o.var["gene_id"].astype(str)).get_indexer(raw.var["gene_id"].astype(str).to_numpy()[gi])
oc = o.obs_names.get_indexer(names)
assert (og >= 0).all() and (oc[keepc] >= 0).all()
Y = sp.csr_matrix(o.X)[np.where(oc >= 0, oc, 0)][:, og].astype(np.float64)
flag = o.obs["glymph_doublet_like"].to_numpy()[np.where(oc >= 0, oc, 0)].astype(bool) if "glymph_doublet_like" in o.obs else np.zeros(len(names), bool)
species = pc.species.reindex(names).to_numpy()
lab = d["lab"].numpy()
fine = d["fine"].numpy() if "fine" in d else lab
parent = d["parent"].numpy() if "parent" in d else np.arange(d["wide"].shape[0])
wide, big, keep = d["wide"].numpy(), d["big"].numpy(), d["keep_c"].numpy()
wide_f = d["wide_f"].numpy() if "wide_f" in d else wide
B, Xc, lv, need = d["Bc"].numpy(), d["Xc"].numpy(), d["lv"], d["need"]
cand = d["rate"].numpy() <= d["delta"] * d["rmax"].numpy()[None, :]
k, G = wide.shape
cat = np.full((k, G), 4, dtype=np.int8)  # 0 silent, 1 overruled (set per entry), 2 not a candidate, 3 above the level, 4 rare above, 5 unused
used = (big & keep)[:, None]
cat[np.broadcast_to(~cand, (k, G))] = 2
cat[cand & (B >= need) & ~wide] = 3
cat[cand & (B < need) & ~wide] = 4
cat[wide] = 0
cat[np.broadcast_to(~used, (k, G))] = 5
names_cat = ["silent", "overruled by the sub-cluster", "not a candidate", "above the level", "rare, above", "cluster unused"]
Xo = X.tocoo()
ok = keepc[Xo.row]
r_, c_, v_ = Xo.row[ok], Xo.col[ok], Xo.data[ok]
cross = np.where(species[r_] == "human", spg[c_] == "mouse", spg[c_] == "human")
r_, c_, v_ = r_[cross], c_[cross], v_[cross]
left = np.asarray(Y[r_, c_]).ravel()
ec = cat[lab[r_], c_].astype(int)
ec[(ec == 0) & ~wide_f[fine[r_], c_]] = 1
tot, L = v_.sum(), left.sum()
print(f"cross-species counts of these cells: {int(tot)}, left {int(round(L))} ({100 * L / tot:.2f}%)")
print("class of the gene in the cell's cluster\tshare of raw\tremoved\tshare of what is left")
for i, nm in enumerate(names_cat):
    m = ec == i
    if m.any():
        print(f"  {nm}\t{100 * v_[m].sum() / tot:.1f}%\t{100 * (1 - left[m].sum() / max(v_[m].sum(), 1)):.1f}%\t{100 * left[m].sum() / max(L, 1):.1f}%")
m = flag[r_]
print(f"cells left alone as doublet-like: {int((flag & keepc).sum())}; they hold {100 * v_[m].sum() / tot:.1f}% of raw and {100 * left[m].sum() / max(L, 1):.1f}% of what is left")
m0 = (ec == 0) & ~flag[r_]
print(f"silent genes in cells not left alone: removed {100 * (1 - left[m0].sum() / max(v_[m0].sum(), 1)):.2f}%")
for sp_name in ("human", "mouse"):
    ms = species[r_] == sp_name
    if not ms.any():
        continue
    line = f"{sp_name} cells: removed {100 * (1 - left[ms].sum() / max(v_[ms].sum(), 1)):.1f}%;"
    for i, nm in enumerate(names_cat):
        m = ms & (ec == i)
        if m.any() and v_[m].sum() / v_[ms].sum() > 0.002:
            line += f" {nm} {100 * v_[m].sum() / v_[ms].sum():.1f}% of raw, removed {100 * (1 - left[m].sum() / max(v_[m].sum(), 1)):.1f}%;"
    print(line)
# by cell: its cross-species share, in thirds
cb = np.bincount(r_, weights=v_, minlength=len(names))
ca = np.bincount(r_, weights=left, minlength=len(names))
fr = cb / np.maximum(np.asarray(X.sum(1)).ravel(), 1)
cells = np.flatnonzero(keepc)
order = cells[np.argsort(fr[cells])]
for nm, sl in (("cleanest third", order[: len(order) // 3]), ("middle third", order[len(order) // 3: 2 * len(order) // 3]), ("dirtiest third", order[2 * len(order) // 3:])):
    inset = np.zeros(len(names), bool); inset[sl] = True
    me = inset[r_]
    line = f"{nm} (cross share {np.median(fr[sl]):.4f}): removed {100 * (1 - ca[sl].sum() / max(cb[sl].sum(), 1)):.1f}%;"
    for i, nm2 in enumerate(names_cat):
        m = me & (ec == i)
        if m.any() and v_[m].sum() / v_[me].sum() > 0.01:
            line += f" {nm2} {100 * v_[m].sum() / v_[me].sum():.1f}% of raw, removed {100 * (1 - left[m].sum() / max(v_[m].sum(), 1)):.1f}%;"
    print(line)
# the genes that hold the most of what is left among the non-silent classes
for i in (3, 4, 2):
    m = ec == i
    if not m.any():
        continue
    gl = np.bincount(c_[m], weights=left[m], minlength=G)
    gr = np.bincount(c_[m], weights=v_[m], minlength=G)
    top = np.argsort(-gl)[:12]
    print(f"{names_cat[i]}: genes holding the most of what is left (left/raw): " + ", ".join(f"{gname[g]}({int(round(gl[g]))}/{int(gr[g])})" for g in top))
