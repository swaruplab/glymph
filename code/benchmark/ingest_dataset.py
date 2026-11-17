#!/usr/bin/env python
"""Turn one downloaded library into the benchmark's standard input.

Every platform arrives in its own format (10x HDF5, BD Rhapsody tables, Parse split-pipe folders,
STARsolo output, lab-specific h5ad). This script reads one row of config/datasets.tsv and writes

    <out>/<name>.h5ad          every barcode with at least one count, integer counts, var with
                               gene_id, gene_name, genome; obs with, where they apply,
                                 unit         barcodes that share a soup (Parse: round 1 well)
                                 pool         barcodes amplified together (Parse: sublibrary)
                                 vendor_cell  the vendor pipeline's (or STARsolo's) cell call
                                 label        the authors' species call, or the tissue loaded in the well
    <out>/<name>.cells.txt     vendor cell calls, one barcode per line (only when they exist)
    <out>/<name>.ingest.json   sizes, thresholds from Glymph's prior step, what was attached
    <out>/qc/<name>.rank.png   barcode rank plot with those thresholds

so that every tool in the benchmark starts from the same matrix.

    python ingest_dataset.py --config config/datasets.tsv --name bd_spleen --data <data root> --out <data root>/std
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph import io as gio  # noqa: E402
from glymph.background import estimate_priors, plot_priors  # noqa: E402

PARSE_BC = "/dfs7/swaruplab/vswarup/bin/ParseBiosciences-Pipeline.1.2.1/splitpipe/barcodes"


def first_column(path, skip_header=True):
    """First comma-separated field of every data line of a BD table (the cell labels)."""
    out, seen_header = [], False
    with gio._open_text(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if skip_header and not seen_header:
                seen_header = True
                continue
            out.append(line[: line.index(",")] if "," in line else line.strip())
    return np.asarray(out)


def merge_rows(a, new_names):
    """Sum the rows of a that share a new name (Parse: the polyT and random-hexamer barcodes of one well)."""
    import anndata as ad

    codes, uniq = pd.factorize(np.asarray(new_names), sort=False)
    G = sp.csr_matrix((np.ones(len(codes), dtype=np.int32), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    X = (G @ sp.csr_matrix(a.X)).tocsr().astype(np.int32)
    return ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(np.asarray(uniq).astype(str), name="barcode")), var=a.var.copy())


def round1_wells(csv):
    t = pd.read_csv(csv)
    return dict(zip(t["sequence"].astype(str), t["well"].astype(str)))


def starsolo_barcodes(folder):
    f = gio._find(gio.Path(folder), "barcodes.tsv")
    return pd.read_csv(f, header=None, sep="\t")[0].astype(str).to_numpy()


def set_genome(a, truth):
    """One genome label per gene. Mixed-species references mark the species in the gene id."""
    gid = a.var["gene_id"].astype(str)
    g = a.var["genome"].astype(str).to_numpy().copy() if "genome" in a.var else np.array([""] * a.n_vars, dtype=object)
    pre = gid.str.extract(r"^(GRCh\d+|hg\d+|GRCm\d+|mm\d+)_")[0]
    g = np.where(pre.notna(), pre.fillna(""), g)
    if truth == "species":
        both = np.array([("hg" in x.lower() or "grch" in x.lower()) and ("mm" in x.lower() or "grcm" in x.lower()) for x in g.astype(str)])
        bare = (g.astype(str) == "") | both
        g = np.where(bare & gid.str.startswith("ENSG").to_numpy(), "GRCh38", g)
        g = np.where(bare & gid.str.startswith("ENSMUSG").to_numpy(), "mm10", g)
    a.var["genome"] = g.astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, help="download root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = pd.read_csv(a.config, sep="\t", dtype=str).fillna("")
    row = cfg[cfg.name == a.name].iloc[0]
    path = os.path.join(a.data, row.path)
    fmt, info = row.format, {"name": a.name, "format": row.format, "source": row.path}
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    vendor = None

    if fmt == "10x_h5":
        ad_ = gio.read_10x_h5(path)
    elif fmt == "10x_h5_gz":
        t = os.path.join(tmp, "raw.h5")
        with gzip.open(path, "rb") as fi, open(t, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        ad_ = gio.read_10x_h5(t)
    elif fmt == "bd_csv":
        ad_ = gio.read_bd_csv(path)
    elif fmt == "bd_long":
        ad_ = gio.read_bd_long(path)
    elif fmt == "bd_mex_zip":
        zipfile.ZipFile(path).extractall(tmp)
        ad_ = gio.read_mtx_dir(tmp, "10x")
    elif fmt == "h5ad":
        ad_ = gio.read_counts(path)
    elif fmt in ("starsolo", "starsolo_parse"):
        ad_ = gio.read_mtx_dir(os.path.join(path, "raw"), "10x")
        vendor = starsolo_barcodes(os.path.join(path, "filtered"))
        if fmt == "starsolo_parse":
            wells = round1_wells(os.path.join(os.path.dirname(os.path.dirname(path)), "whitelist", "round1_wells.csv"))
            def to_well(b):
                parts = np.char.rpartition(np.asarray(b).astype(str), "_")  # round 3 _ round 2 | _ | round 1
                return np.char.add(np.char.add(parts[:, 0], "_"), np.array([wells[x] for x in parts[:, 2]])), np.array([wells[x] for x in parts[:, 2]])
            names, _ = to_well(ad_.obs_names.to_numpy())
            ad_ = merge_rows(ad_, names)
            ad_.obs["unit"] = np.char.rpartition(ad_.obs_names.to_numpy().astype(str), "_")[:, 2]
            vendor = np.unique(to_well(vendor)[0])
            info["round1_merged"] = True
    elif fmt == "parse_dge":
        ad_ = gio.read_mtx_dir(path, "parse")
        names = ad_.obs_names.to_series().astype(str)
        ad_.obs["pool"] = names.str.extract(r"__(s\d+)$")[0].fillna("s1").to_numpy()
        if "bc1_well" in ad_.obs:
            ad_.obs["unit"] = ad_.obs["bc1_well"].astype(str).to_numpy()
        else:
            ad_.obs["unit"] = ("w" + names.str.split("_").str[0]).to_numpy()
        if "sample" in ad_.obs:
            ad_.obs["label"] = ad_.obs["sample"].astype(str).to_numpy()
        ad_.obs = ad_.obs[[c for c in ("unit", "pool", "label") if c in ad_.obs]]
    elif fmt == "igvf_parse":
        ad_ = gio.read_counts(path)
        bc = ad_.obs_names.to_series().astype(str).str.split("_").str[0]
        seg = [bc.str.slice(i, i + 8) for i in (0, 8, 16)]
        wells = round1_wells(os.path.join(PARSE_BC, "bc_data_n99_v5.csv"))
        # the round 1 segment is the one whose barcodes all come from the kit's round 1 list
        frac = [float(s_.isin(list(wells)).mean()) for s_ in seg]
        nuni = [int(s_.nunique()) for s_ in seg]
        k = int(np.argmax([f - 1e-6 * n for f, n in zip(frac, nuni)]))
        info["round1_segment"] = {"index": k, "fraction_in_list": frac, "distinct": nuni}
        if frac[k] < 0.99:
            raise SystemExit(f"{a.name}: no barcode segment matches the round 1 list ({frac})")
        ad_.obs = pd.DataFrame({"unit": seg[k].map(wells).to_numpy()}, index=ad_.obs_names)
    else:
        raise SystemExit(f"unknown format {fmt}")

    set_genome(ad_, row.truth)
    names = ad_.obs_names.to_numpy().astype(str)

    if row.cells and row.cells.endswith((".h5", ".hdf5")):  # a Cell Ranger filtered matrix: its barcodes
        import h5py

        with h5py.File(os.path.join(a.data, row.cells), "r") as f:
            vendor = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["matrix/barcodes"][:]])
    elif row.cells:
        vendor = first_column(os.path.join(a.data, row.cells))
    if row.labels:
        lp = os.path.join(a.data, row.labels)
        if a.name == "jurkat_32d":
            t = pd.read_csv(lp, sep="\t")
            lab = pd.Series(t["org"].astype(str).to_numpy(), index=t["barcode"].astype(str) + "-1")
            vendor = lab.index.to_numpy()
        elif a.name == "neat_barnyard":
            t = pd.read_csv(lp, sep="\t")
            lab = pd.Series(t["RNA_species"].astype(str).to_numpy(), index=t["cell_id"].astype(str))
            vendor = t.loc[t["RNA_pass"].astype(str).str.upper() == "TRUE", "cell_id"].astype(str).to_numpy()
        elif row.format == "starsolo":
            t = pd.read_csv(lp)
            lab = pd.Series(t["species"].astype(str).to_numpy(), index=t.iloc[:, 1].astype(str))
        elif row.format == "igvf_parse":
            t = pd.read_csv(lp)
            t = t[t["Experiment"] == "igvf_012"]
            well_tissue = dict(zip(t["bc1_well"].astype(str), t["Tissue"].astype(str)))
            ad_.obs["label"] = ad_.obs["unit"].map(well_tissue).fillna("").to_numpy()
            info["wells_with_tissue"] = int(ad_.obs["unit"].isin(list(well_tissue)).mean() * 1e4) / 1e4
            lab = None
        else:
            raise SystemExit(f"{a.name}: no rule for the labels file")
        if lab is not None:
            lab = lab[~lab.index.duplicated()]
            ad_.obs["label"] = lab.reindex(names).fillna("").to_numpy()
            info["labels_matched"] = int((ad_.obs["label"] != "").sum())
            info["labels_in_file"] = int(len(lab))
    if vendor is not None:
        ad_.obs["vendor_cell"] = np.isin(names, np.asarray(vendor).astype(str))
        info["vendor_cells_in_file"] = int(len(vendor))
        info["vendor_cells_matched"] = int(ad_.obs["vendor_cell"].sum())

    X = sp.csr_matrix(ad_.X).astype(np.int32)
    ad_.X = X
    N = np.asarray(X.sum(1)).ravel()
    os.makedirs(os.path.join(a.out, "qc"), exist_ok=True)
    for c in ad_.obs.columns:
        if ad_.obs[c].dtype == object:
            ad_.obs[c] = ad_.obs[c].astype(str)
    ad_.write_h5ad(os.path.join(a.out, f"{a.name}.h5ad"), compression="gzip", compression_opts=1)
    if "vendor_cell" in ad_.obs:
        with open(os.path.join(a.out, f"{a.name}.cells.txt"), "w") as f:
            f.write("\n".join(names[ad_.obs["vendor_cell"].to_numpy()]) + "\n")

    info.update({
        "n_barcodes": int(X.shape[0]), "n_genes": int(X.shape[1]), "n_counts": int(N.sum()), "nnz": int(X.nnz),
        "depth_quantiles": {q: float(np.quantile(N, float(q))) for q in ("0.5", "0.9", "0.99", "0.999")},
        "genomes": {k: int(v) for k, v in ad_.var["genome"].value_counts().items()},
        "obs_columns": list(ad_.obs.columns),
    })
    for c in ("unit", "pool", "label"):
        if c in ad_.obs:
            vc = ad_.obs[c].value_counts()
            info[f"n_{c}"] = int(len(vc))
            info[f"{c}_top"] = {str(k): int(v) for k, v in vc.head(8).items()}
    try:
        pri = estimate_priors(X)
        d = pri.diagnostics
        info["priors"] = {k: (float(d[k]) if isinstance(d[k], (int, float, np.integer, np.floating)) else str(d[k]))
                          for k in ("threshold_method", "cell_threshold", "n_cells_prior", "empty_level", "empty_max",
                                    "n_background_barcodes", "n_analyzed") if k in d}
        if "vendor_cell" in ad_.obs:
            v = ad_.obs["vendor_cell"].to_numpy()
            info["priors"]["vendor_cells_above_threshold"] = int((N[v] >= d["cell_threshold"]).sum())
            info["priors"]["vendor_median_depth"] = float(np.median(N[v])) if v.any() else 0.0
        plot_priors(X, pri, os.path.join(a.out, "qc", f"{a.name}.rank.png"))
    except Exception as e:  # the readiness check must not hide a matrix that was read correctly
        info["priors_error"] = f"{type(e).__name__}: {e}"
    with open(os.path.join(a.out, f"{a.name}.ingest.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
