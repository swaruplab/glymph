# Data locations

Large data stays on the compute host. Sizes and checksums measured 2026-11-16.

## Inputs

Tier 1 raw 10x matrices, `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm/` (957 MB in total, downloaded 2026-11-16 from the URLs in `config/benchmark_tier1.tsv`).

| Library | File | md5 |
|---|---|---|
| hgmm_1k_v3 | hgmm_1k_v3_raw_feature_bc_matrix.h5 | e1c70edaabd92cef1a0d777cdcc9c311 |
| hgmm_5k_v3 | hgmm_5k_v3_raw_feature_bc_matrix.h5 | b7976cb134ca242f048b5fc3603e8083 |
| hgmm_6k_v2 | hgmm_6k_raw_gene_bc_matrices_h5.h5 | f725e3f6e722020d6d3b601961e85a5a |
| hgmm_10k_v3 | hgmm_10k_v3_raw_feature_bc_matrix.h5 | 42fa0c34cc1d154ff309ca2e8ad69f62 |
| hgmm_12k_v2 | hgmm_12k_raw_gene_bc_matrices_h5.h5 | 916c07c7659b06fc673fc6aeedba5693 |
| hgmm_20k_v31_HT | 20k_hgmm_3p_HT_nextgem_Chromium_X_raw_feature_bc_matrix.h5 | a0c847cb1716e6696f183c20c7854663 |
| hgmm_10k_gemx_v4 | 10k_hgmm_3p_gemx_Multiplex_count_raw_feature_bc_matrix.h5 | b5868f79f12c617d1184d63964571ed4 |

Everything below sits under `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/` and was downloaded on 2026-11-16 straight onto HPC3 through the transfer node. Every file and its source link is in `config/datasets_manifest.tsv`; the libraries built from them are in `config/datasets.tsv`. Sizes measured 2026-11-16 (21:20 EST), read files still arriving.

| Folder | What | Size |
|---|---|---|
| `kidney_strainmix/` | Tier 2: mouse kidney from three strains, GSE218853, five raw and five filtered 10x matrices; the authors' per-cell truth is in the project's `data/kidney_strainmix/` | 1.2 GB |
| `hgmm_holdout/` | eight more 10x human + mouse mixtures (5' and 3' chemistries), never used for tuning | 1.1 GB |
| `neat_barnyard/`, `jurkat_32d/` | nuclei barnyard (GSE178707) and Jurkat + 32D (GSM4228200), raw matrices and the authors' species calls | 0.1 GB |
| `xenograft_skin/` | six raw matrices, mouse skin with or without a human immune system, GSE240009 | 0.1 GB |
| `pfc_nuclei/` | one human prefrontal cortex nuclei library, GSE168408 (no truth) | 0.1 GB |
| `rhapsody/` | BD Rhapsody: demo PBMC (MEX), mouse splenocytes (GSE261975), five ileum cartridges (GSE348009), two PBMC donors (GSE310996), liver, lung and prostate (Zenodo 12165000), and the human + mouse mixture GSE265962 as reads (`rock_roi/fastq/`, 14 GB) with the authors' species labels | about 26 GB with reads |
| `parse/` | Parse Evercode: eight IGVF sublibraries of plate igvf_012 with the plate's well map, bone marrow GSE252604, fracture callus GSE292578 (called cells only, unusable), and the human + mouse mixture PRJEB67549 as reads (`PRJEB67549/fastq/`, 22 GB) | about 29 GB with reads |
| `std/` | the benchmark's standard inputs: one `.h5ad`, cell list, summary and rank plot per library (`code/benchmark/ingest_dataset.py`) | 7.5 GB |

`hpc3:/dfs7/swaruplab/vswarup/compute/glymph/ref/star_GRCh38_mm10_2020A_2710a_sparse2/`: STAR 2.7.10a index of the combined human + mouse genome (10x GRCh38-and-mm10-2020-A, sparse suffix array), 33 GB. Built for the two read-based mixtures; can be deleted once they are aligned, or kept for more.

`hpc3:/dfs7/swaruplab/vswarup/compute/glymph/dev/`: development copy of the code and its trial outputs, 13 GB and growing. Only the scores matter; safe to delete when development ends.

## Run outputs

`hpc3:/dfs7/swaruplab/vswarup/compute/glymph/20261116-1539_glymph-hgmm-bench/` (44 GB at the end of the run; `/dfs7` was 86% full).

| Folder under `out/` | What | Size | Keep? |
|---|---|---|---|
| `glymph/`, `glymph_fpr0/`, `glymph_fpr1/` | Glymph 0.2.0 at a false-positive budget of 0.5% (default), 0 and 1%: `.h5ad`, report, rank plot per library | 4.2 GB | scored outputs |
| `glymph_v01/`, `glymph_v0/` | Glymph 0.1.0 and 0.0.1 | 2.8 GB | scored outputs, kept for the version history |
| `glymph_4gpu/` | the seven libraries from the 0.2.0 run on one 4-GPU node | 1.2 GB | timing run; duplicates `glymph/` |
| `cellbender/` | CellBender outputs per library (raw and filtered `.h5`, posterior, logs) | 4.4 GB | filtered `.h5` are scored |
| `soupx/`, `decontx/`, `decontx_bg/` | corrected matrices (fractional counts, so large) | 6.5 GB | scored outputs |
| `scar/`, `ambidose/` | corrected matrices | 0.8 GB | scored outputs |
| `exp_*/` (13 folders) | development experiments: parsimony prior, longer training, fixed swap fraction, calibration variants | about 17 GB | only their reports and scores matter |
| `prep/` | one v3-layout raw matrix and one cell list per library, shared by the baselines | 2.7 GB | regenerated in under a minute per library |
| `*_failed/`, `glymph_4gpu_v0_run/` | outputs of failed attempts and of the first 4-GPU run | about 2 GB | can go |
| `metrics/`, `metrics_exp/`, `metrics_v02*/`, `deliverables/`, `diag/`, `envs/` | tables, per-cell tables, figures, version files | small | copied to `runs/<run_id>/` |

Nothing has been deleted. Removing `exp_*`, `prep/`, the failed attempts, `glymph_4gpu_v0_run/` and `glymph_4gpu/` would free about 25 GB.

## Environments built for the benchmark

Under `hpc3:/dfs7/swaruplab/vswarup/bin/miniconda_envs/`: `glymph_bench_r` (1.0 GB), `glymph_bench_scar` (2.4 GB), `glymph_bench_ambidose` (0.3 GB). Versions are in `runs/20261116-1539_glymph-hgmm-bench/*.versions.txt`.

## Kidney strain mixture: alignment files and spliced/unspliced counts (added 2026-11-18)

For the spliced/unspliced measurement in Glymph 0.4.0 development (run `20261118-1245_glymph-040-dev`).

| What | Where | Size | Source |
|---|---|---|---|
| Cell Ranger alignment files of the five libraries | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix/bam/{nuc2,nuc3,rep1,rep2,rep3}_possorted_genome_bam.bam` | 20.1, 29.4, 34.4, 43.4, 16.6 GB (144 GB) | authors' submitted files in the SRA public bucket (`sra-pub-src-1` and `sra-pub-src-2`), fetched 2026-11-18 by `code/benchmark/50_download_kidney_bams.sh`; md5 checked against the bucket's ETag where it is a plain md5 |
| Molecules per gene and barcode: unspliced, spliced, ambiguous | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/splice/kidney_{nuc2,nuc3,rep3}.{unspliced,spliced,ambiguous}.npz` | under 1 GB | `code/benchmark/51_bam_splice_counts.py` (reads classified by position against the 10x mm10-2020-A annotation) |
| Standardised nuclei libraries with `layers["unspliced"]` | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/std_splice/kidney_{nuc2,nuc3}.h5ad` | under 2 GB | `code/benchmark/52_add_splice_layers.py` |

Only the two nuclei libraries carry splicing information. The three whole-cell libraries (rep1, rep2, rep3; 94 GB of alignment files) were aligned without introns: 0.4 to 0.7% of their molecules are unspliced everywhere, so their files are of no further use here and can be deleted once that is approved.

## Outputs of the 0.4.0 development and confirmation runs (added 2026-11-19)

Decontaminated matrices and reports, one folder per tool or variant, kept on HPC3. Score tables are in each run folder under `runs/`.

| What | Where | Size |
|---|---|---|
| Development rounds M1 to M8 and the splicing rounds (variants of Glymph 0.4.0.dev on 23 to 26 libraries; current ones in `out/`, earlier ones in `out_old/`; test outputs with the silent-test dump in `out_dbg/`) | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev/` | 109 GB |
| Glymph 0.4.0rc1, 54 libraries, two passes | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/20261118-2004_glymph-040rc1-bench/out/{glymph,glymph_rep2}/` | 15 GB |
| Glymph 0.4.0rc2, 54 libraries, two passes | `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/20261118-2355_glymph-040rc2-bench/out/{glymph,glymph_rep2}/` | 15 GB |

The development folder can go once 0.4.0 is settled (it holds nothing that the two confirmation runs and the tables do not cover, except the dumps used for diagnosis). Nothing has been deleted; deletion needs a yes.

## Scratch elsewhere

`studio-work:~/LocalDocuments/compute/_dev_glymph/`: a copy of `src`, `tests`, `code`, `config` used for CPU unit tests (a few MB). Safe to delete.
