# Quickstart

## Install

```bash
git clone https://github.com/swaruplab/glymph.git
cd glymph
pip install -e .
```

Requirements: Python 3.10 or newer, PyTorch 2.2 or newer, numpy, scipy, pandas, h5py, anndata. One CUDA GPU is recommended;
a typical library takes one to four minutes there. Glymph also runs on CPU.

## Run one library

```bash
glymph run --input raw_feature_bc_matrix.h5 --output sample.h5ad --plot sample.rank.png
```

The input must be the **raw** matrix with every barcode, not the filtered one: Glymph needs the empty partitions.

Inputs it reads: 10x `.h5`, MEX folders (Cell Ranger, STARsolo, BD Rhapsody), BD Rhapsody tables, Parse split-pipe folders,
and `.h5ad`.

## Run many libraries on several GPUs

```bash
glymph batch --manifest libs.tsv --outdir out --devices cuda:0,cuda:1,cuda:2,cuda:3
```

One library per GPU at a time.

## Platform structure

For split-pool data, pass an `.h5ad` whose `obs` says which barcodes share ambient RNA and which were amplified together:

```bash
glymph run --input parse_sublibrary.h5ad --output clean.h5ad --unit-key unit --pool-key pool
```

## What comes out

| Where | What |
|---|---|
| `X` | cleaned integer counts for called cells |
| `layers["raw"]` | the input counts |
| `obs["glymph_background_fraction"]` | the share of each cell's counts removed |
| `obs["glymph_cell_probability"]` | probability that the barcode holds a cell |
| `obs["glymph_doublet_like"]` | cells left alone because they express their cluster's silent genes themselves (a rare cell type or a doublet) |
| `obs["glymph_background_match"]`, `obs["glymph_silent_share"]` | for every cell that was examined: how closely its counts on those genes follow the background profile, and their share of the cell |
| `obsm["X_glymph"]` | the embedding |
| `sample.report.json` | timings, thresholds, the measured amounts and their diagnostics |

## Options worth knowing

| Option | Default | Meaning |
|---|---|---|
| `--fpr` | 0.004 | False-positive budget: trades removal against counts kept |
| `--doublet-match` | 0.5 | A cell with a large excess on its cluster's silent genes is left alone when those counts match the background profile less than this |
| `--doublet-share-max` | 0.2 | ... or when they are more than this share of the cell |
| `--spare-doublets` | off | Leave every such cell alone, whatever its profile |
| `--no-silent-broad` | | Use only genes abundant enough to test one by one |
| `--calib-seeds` | 3 | Clusterings tried for the measurement; the middle estimate is kept |
| `--device` | auto | `cuda:0`, `cpu`, ... |
