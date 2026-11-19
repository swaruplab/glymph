"""Readers for raw (unfiltered) count matrices.

Every reader returns an AnnData with barcodes in rows, genes in columns, a CSR
matrix of integer counts, and var columns gene_id, gene_name, feature_type, genome.

Formats:
  10x HDF5 (Cell Ranger v3+ and the legacy v2 per-genome layout).
  MEX directories in 10x layout (Cell Ranger, STARsolo, BD Rhapsody MEX export).
  BD Rhapsody tables: the dense *_MolsPerCell[_Unfiltered].csv and the long
      *_Expression_Data[_Unfiltered].st.txt.
  Parse split-pipe DGE folders (cells x genes, all_genes.csv, cell_metadata.csv),
      including the prefixed copies deposited on GEO.
  .h5ad.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _dec(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _finalize(X, barcodes, var: pd.DataFrame, source: str, drop_empty: bool = True, integer: bool = True) -> ad.AnnData:
    X = sp.csr_matrix(X)
    if integer:
        if not np.issubdtype(X.dtype, np.integer):
            if X.nnz and not np.allclose(X.data, np.round(X.data)):
                raise ValueError("Glymph needs raw integer counts; found non-integer values.")
        X = X.astype(np.int32)
    else:  # another tool's output, kept exactly as written (fractional or negative values are scored)
        X = X.astype(np.float64)
    X.sum_duplicates()
    X.sort_indices()
    barcodes = np.asarray(barcodes).astype(str)
    if drop_empty:
        keep = np.diff(X.indptr) > 0
        X, barcodes = X[keep], barcodes[keep]
    var = var.reset_index(drop=True).copy()
    for col, default in (("gene_name", None), ("feature_type", "Gene Expression"), ("genome", "")):
        if col not in var:
            var[col] = var["gene_id"] if default is None else default
    var.index = ad.utils.make_index_unique(pd.Index(var["gene_id"].astype(str)))
    out = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")), var=var)
    out.uns["glymph_source"] = str(source)
    return out


def read_10x_h5(path, integer: bool = True) -> ad.AnnData:
    """Cell Ranger raw_feature_bc_matrix.h5 (v3+) or raw_gene_bc_matrices_h5.h5 (v2)."""
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            g = f["matrix"]
            shape = tuple(int(s) for s in g["shape"][:])
            X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape).T.tocsr()
            barcodes = _dec(g["barcodes"][:])
            feat = g["features"]
            var = pd.DataFrame({"gene_id": _dec(feat["id"][:]), "gene_name": _dec(feat["name"][:])})
            if "feature_type" in feat:
                var["feature_type"] = _dec(feat["feature_type"][:])
            if "genome" in feat:
                var["genome"] = _dec(feat["genome"][:])
        else:
            # Cell Ranger v2: one top-level group per genome, each with its own matrix.
            mats, vars_, barcodes = [], [], None
            for genome in sorted(f.keys()):
                g = f[genome]
                shape = tuple(int(s) for s in g["shape"][:])
                m = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape).T.tocsr()
                bc = _dec(g["barcodes"][:])
                if barcodes is None:
                    barcodes = bc
                elif not np.array_equal(bc, barcodes):
                    idx = pd.Index(bc).get_indexer(barcodes)
                    if (idx < 0).any():
                        raise ValueError(f"{path}: genome groups have different barcode sets")
                    m = m[idx]
                mats.append(m)
                vars_.append(
                    pd.DataFrame(
                        {
                            "gene_id": _dec(g["genes"][:]),
                            "gene_name": _dec(g["gene_names"][:]),
                            "feature_type": "Gene Expression",
                            "genome": genome,
                        }
                    )
                )
            X = sp.hstack(mats, format="csr")
            var = pd.concat(vars_, ignore_index=True)
    if "feature_type" in var:
        keep = (var["feature_type"] == "Gene Expression").to_numpy()
        X, var = X[:, keep], var.loc[keep]
    return _finalize(X, barcodes, var, path, integer=integer)


def _find(p: Path, *names):
    """First match for each name in turn: exact, gzipped, then any file ending in the name
    (GEO deposits prefix every file with the series accession)."""
    for n in names:
        for ext in ("", ".gz"):
            if (p / (n + ext)).exists():
                return p / (n + ext)
    for n in names:
        for ext in ("", ".gz"):
            hits = sorted(p.glob("*" + n + ext))
            if hits:
                return hits[0]
    return None


def _open_text(path):
    import gzip

    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _bd_feature_table(names) -> pd.DataFrame:
    """BD tables name genes by symbol; AbSeq and Sample Tag features carry a |pAbO or |stAbO suffix."""
    names = pd.Series(np.asarray(names).astype(str))
    kind = np.where(names.str.contains(r"\|(?:pAbO|stAbO)$|^SampleTag", regex=True), "Antibody Capture", "Gene Expression")
    return pd.DataFrame({"gene_id": names, "gene_name": names, "feature_type": kind})


def read_bd_csv(path, integer: bool = True) -> ad.AnnData:
    """BD Rhapsody dense table (*_RSEC_MolsPerCell[_Unfiltered].csv, also DBEC): comment lines
    starting with #, a header 'Cell_Index,<features>', then one row of counts per cell label."""
    rows, cols, vals, cells, header = [], [], [], [], None
    with _open_text(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if header is None:
                header = line.rstrip("\n").split(",")
                if header[0] != "Cell_Index":
                    raise ValueError(f"{path}: expected a BD table starting with Cell_Index, found {header[0]!r}")
                continue
            k = line.index(",")
            v = np.fromstring(line[k + 1:], dtype=np.float64, sep=",")
            if v.size != len(header) - 1:
                raise ValueError(f"{path}: row for cell {line[:k]} has {v.size} values, header has {len(header) - 1}")
            nz = np.flatnonzero(v)
            rows.append(np.full(nz.size, len(cells), dtype=np.int64))
            cols.append(nz)
            vals.append(v[nz])
            cells.append(line[:k])
    if header is None:
        raise ValueError(f"{path}: no header line found")
    var = _bd_feature_table(header[1:])
    X = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cells), len(header) - 1)
    )
    keep = (var["feature_type"] == "Gene Expression").to_numpy()
    return _finalize(X[:, keep], np.asarray(cells), var.loc[keep], path, integer=integer)


def read_bd_long(path, value: str = "auto", integer: bool = True) -> ad.AnnData:
    """BD Rhapsody long table (*_Expression_Data[_Unfiltered].st): one row per cell label and gene.
    value: the count column; 'auto' takes RSEC_Adjusted_Molecules, the pipeline's UMI-corrected
    count before BD's own noise filter (DBEC), and falls back to Raw_Molecules."""
    head = pd.read_csv(path, sep="\t", comment="#", nrows=5)
    gene_col = "Gene" if "Gene" in head else "Bioproduct"
    if value == "auto":
        value = next(c for c in ("RSEC_Adjusted_Molecules", "Raw_Molecules", "DBEC_Adjusted_Molecules") if c in head)
    df = pd.read_csv(path, sep="\t", comment="#", usecols=["Cell_Index", gene_col, value],
                     dtype={"Cell_Index": str, gene_col: str})
    df = df[df[value] > 0]
    ci, cells = pd.factorize(df["Cell_Index"], sort=False)
    gi, genes = pd.factorize(df[gene_col], sort=True)
    X = sp.csr_matrix((df[value].to_numpy(), (ci, gi)), shape=(len(cells), len(genes)))
    var = _bd_feature_table(np.asarray(genes))
    keep = (var["feature_type"] == "Gene Expression").to_numpy()
    return _finalize(X[:, keep], np.asarray(cells).astype(str), var.loc[keep], path, integer=integer)


def read_parse(mtx, genes, cells, integer: bool = True) -> ad.AnnData:
    """Parse split-pipe DGE matrix (cells x genes) with its gene and cell tables. The cell table's
    columns (bc_wells, sample, bc1_well, ...) are kept in obs; bc1_well is the round 1 well."""
    from scipy.io import mmread

    genes = pd.read_csv(genes)
    cells = pd.read_csv(cells)
    M = mmread(str(mtx)).tocsr()
    if M.shape == (len(genes), len(cells)) and len(genes) != len(cells):
        M = M.T.tocsr()
    if M.shape != (len(cells), len(genes)):
        raise ValueError(f"{mtx}: matrix is {M.shape}, tables give {len(cells)} cells and {len(genes)} genes")
    gid = genes["gene_id"] if "gene_id" in genes else genes.iloc[:, 0]
    var = pd.DataFrame(
        {
            "gene_id": gid.astype(str),
            "gene_name": genes.get("gene_name", gid).astype(str),
            "genome": genes.get("genome", pd.Series([""] * len(genes))).astype(str),
        }
    )
    key = next((c for c in ("bc_wells", "bc_index") if c in cells), cells.columns[0])  # GEO copies may lead with a row number
    names = ad.utils.make_index_unique(pd.Index(cells[key].astype(str)))
    out = _finalize(M, names.to_numpy(), var, mtx, drop_empty=False, integer=integer)
    meta = cells.copy()
    meta.index = names
    out.obs = meta
    keep = np.diff(out.X.indptr) > 0
    return out[keep].copy()


def read_mtx_dir(path, layout: str = "auto", integer: bool = True) -> ad.AnnData:
    """MEX directory. layout: '10x' (genes x barcodes; also BD Rhapsody MEX), 'parse', or 'auto'."""
    from scipy.io import mmread

    p = Path(path)
    if layout == "auto":
        layout = "parse" if _find(p, "all_genes.csv", "DGE.mtx") is not None else "10x"
    if layout == "parse":
        return read_parse(
            _find(p, "count_matrix.mtx", "DGE.mtx"),
            _find(p, "all_genes.csv", "genes.csv"),
            _find(p, "cell_metadata.csv", "cell.csv"),
            integer=integer,
        )
    mtx = _find(p, "matrix.mtx")
    bc = pd.read_csv(_find(p, "barcodes.tsv"), header=None, sep="\t")[0].astype(str).to_numpy()
    ft = pd.read_csv(_find(p, "features.tsv", "genes.tsv"), header=None, sep="\t")
    var = pd.DataFrame({"gene_id": ft[0].astype(str), "gene_name": ft[1 if ft.shape[1] > 1 else 0].astype(str)})
    if ft.shape[1] > 2:
        var["feature_type"] = ft[2].astype(str)
    M = mmread(str(mtx))
    if M.shape == (len(var), len(bc)):  # 10x, STARsolo: features x barcodes
        M = M.T.tocsr()
    elif M.shape == (len(bc), len(var)):  # some exports are already barcodes x features
        M = M.tocsr()
    else:
        raise ValueError(f"{mtx}: matrix is {M.shape}, but there are {len(bc)} barcodes and {len(var)} features")
    if "feature_type" in var:
        keep = (var["feature_type"] == "Gene Expression").to_numpy()
        if keep.any():
            M, var = M[:, keep], var.loc[keep]
    return _finalize(M, bc, var, path, integer=integer)


def read_counts(path, layout: str = "auto", integer: bool = True) -> ad.AnnData:
    """Dispatch on the path: .h5 (10x), .h5ad, a MEX directory, or a BD Rhapsody table.

    integer=False reads another tool's output without forcing integer counts."""
    p = Path(path)
    if p.is_dir():
        return read_mtx_dir(p, layout, integer=integer)
    if p.suffix == ".h5ad":
        a = ad.read_h5ad(p)
        var = a.var.copy()
        if "gene_id" not in var:  # scanpy-style files keep the ids in gene_ids and symbols in the index
            var["gene_id"] = var["gene_ids"].astype(str) if "gene_ids" in var else var.index.astype(str)
        out = _finalize(a.X, a.obs_names.to_numpy(), var, path, drop_empty=False, integer=integer)
        out.obs = a.obs.copy()
        if "unspliced" in a.layers:  # molecules with intronic sequence, beside X (splice.py)
            out.layers["unspliced"] = sp.csr_matrix(a.layers["unspliced"]).astype(np.int32)
        keep = np.diff(out.X.indptr) > 0
        return out[keep].copy()
    if p.suffix in (".h5", ".hdf5"):
        return read_10x_h5(p, integer=integer)
    stem = p.name[:-3] if p.name.endswith(".gz") else p.name
    if stem.endswith(".csv"):
        return read_bd_csv(p, integer=integer)
    if stem.endswith((".st", ".st.txt")):
        return read_bd_long(p, integer=integer)
    raise ValueError(f"Do not know how to read {path}")


def read_barcode_list(path) -> np.ndarray:
    """One barcode per line (plain or .gz); used for vendor cell calls."""
    return pd.read_csv(path, header=None, sep="\t")[0].astype(str).to_numpy()
