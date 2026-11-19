# Benchmark

Glymph is tested on the datasets and metrics that the published ambient-removal benchmarks use, so its numbers can be placed next to theirs. CellBender at defaults is the baseline to beat; SoupX, DecontX, scAR and AmbiDose run beside it.

Sources: Janssen et al. 2023 (Genome Biology), Cargnelli et al. 2026 (Frontiers in Bioinformatics), Schroeder et al. 2026 (bioRxiv), Nie et al. 2026 (Nature Communications), Fleming et al. 2023 (CellBender), and the CellSweep and AmbiDose preprints. Details and links are in `reports/Ambient RNA removal beyond CellBender.md`.

Results, job states and the decision log are on the project dashboard (link in `runs/<run_id>/RUN.md`) and in `runs/<run_id>/`. First run, `runs/20261116-1539_glymph-hgmm-bench/`: Glymph 0.2.0 removes 93.3% of cross-species counts against CellBender's 94.3%, with lower native loss and per-cell error and about a thirteenth of the wall time; by the criteria at the end of this file that is level with CellBender, not a win.

## Tier 1: 10x human + mouse mixtures (first run 2026-11-16)

`config/benchmark_tier1.tsv`. Seven raw matrices from the 10x public site, about 1 GB, at `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm`.

| Dataset | Chemistry | Used in |
|---|---|---|
| hgmm_1k_v3, hgmm_5k_v3, hgmm_10k_v3 | 10x 3' v3 | Schroeder size series |
| hgmm_6k_v2, hgmm_12k_v2 | 10x 3' v2 | Schroeder size series; hgmm12k is the CellBender paper's mixture |
| hgmm_20k_v31_HT | 10x 3' v3.1 HT | Schroeder size series |
| hgmm_10k_gemx_v4 | 10x GEM-X 3' v4 | AmbiDose held-out mixture |

Two things are not confirmed: which chemistry version of each "hgmm Nk" set Schroeder et al. used (their text does not say), and that the GEM-X file here (`10k_hgmm_3p_gemx_Multiplex`) is the same library AmbiDose used.

Published reference values on these mixtures (Schroeder et al., mean of five, 1k excluded): cross-species counts removed 94.0% for CellBender 0.3.0, 95.6% scAR, 70.9% SoupX, 53.2% DecontX.

### Tools

| Tool | Version | Environment on HPC3 | How it is run |
|---|---|---|---|
| Glymph | this repository | `scanpy` (existing), code from the run folder | `glymph run`, defaults |
| CellBender | 0.3.2 | `/dfs7/swaruplab/vswarup/bin/cellbender_fix` (lab launcher) | defaults, `--checkpoint-mins 20` |
| SoupX | 1.6.2 | `glymph_bench_r` | Seurat clusters, `autoEstCont`, `adjustCounts` |
| DecontX | decontX 1.8.0 | `glymph_bench_r` | `decontX(x)`; and `decontX(x, background = other droplets)` as `decontx_bg` |
| scAR | 0.7.0 | `glymph_bench_scar` | tutorial: `setup_anndata(prob = 0.995)`, 200 epochs, batch 64 |
| AmbiDose | 0.5.15 | `glymph_bench_ambidose` | `ambidose denoise --cells-only`, 10x MTX output |

Things that were not obvious and cost a rerun each:
- CellBender 0.3.2 on HPC3 dies at its checkpoint save unless launched through `cellbender_fix/cellbender_safe.py`.
- `pip install scar` installs an unrelated cloud tool. scAR is `github.com/Novartis/scar`.
- AmbiDose 0.5.15 fails writing `.h5ad` on most libraries; use `--output-format 10x-mtx`. On libraries of about 9,000 cells or more it fails inside scanpy 1.11 (a sparse array type scanpy cannot read), with scipy 1.17 and 1.14 and with either graph option, so it is scored on three of seven libraries.
- The user `pip.conf` on HPC3 adds an NVIDIA index that compute nodes cannot resolve, and `~/.local` packages leak into new environments. The build script sets `PIP_CONFIG_FILE=/dev/null` and `PYTHONNOUSERSITE=1`.
- CellSweep is not included: no public package or repository could be found.

SoupX, DecontX, scAR and AmbiDose need cell calls. All four get the same list: every barcode at or above the order-of-magnitude count threshold of the raw matrix (`prep_inputs.py`). The evaluation cells are a subset of that list.

### Metrics (`species_mix_metrics.py`)

| Metric | Meaning | From |
|---|---|---|
| `cross_species_removed` | share of known cross-species counts removed | Schroeder "sensitivity" |
| `median_cell_removal` | per-cell median of the same | AmbiDose |
| `own_retained`, `native_loss` | same-species counts kept; native counts removed beyond the expected same-species background | Cargnelli, Schroeder |
| `same_species_ratio` | same-species removed / expected (ideal 1) | Schroeder |
| `n_above_raw`, `added_counts`, `n_negative`, `n_noninteger` | count integrity | Schroeder |
| `contam_rmsle`, `contam_kendall` | per-cell contamination, estimate vs truth | Janssen, Nie |
| wall time, peak RAM, peak GPU memory | from `/usr/bin/time -v` and the Glymph report | all |

Evaluation cells are picked from the raw matrix only (above the order-of-magnitude threshold, at least 90% one species), so every tool is scored on the same barcodes. Tool outputs are read as written: fractional or negative values are not rounded away, they are counted.

### Jobs

Run from the run directory on HPC3. Every GPU job asks for an A30 so run times compare.

| Script | What | Resources |
|---|---|---|
| `00_download_tier1.sh` | `wget` onto HPC3 through `hpc3-xfer` | none |
| `03_prep_inputs.sbatch` | one v3-layout raw file and one cell list per library, for the baselines | 7 x (2 CPU, 24 GB) |
| `05_build_envs.sbatch` | builds the three baseline environments inside a compute job | 3 x (4 CPU, 24 GB) |
| `10_glymph_array.sbatch` | Glymph, one library per task, output in `out/glymph` | 7 x (1 A30, 4 CPU, 48 GB, 1 h limit), `vswarup_lab_gpu`, `gpu` |
| `11_glymph_multigpu.sbatch` | all 7 libraries on one 4-GPU node (`glymph batch`) | 4 A30, 16 CPU, 128 GB, 1 h limit |
| `12_glymph_variant_array.sbatch` | any Glymph configuration, output in `out/<TAG>` | as `10_` |
| `20_cellbender_array.sbatch` | CellBender 0.3.2 defaults | 7 x (1 A30, 4 CPU, 64 GB, 8 h limit) |
| `21_r_baselines_array.sbatch` | SoupX, DecontX, DecontX with background | 7 x (4 CPU, 64 GB, 6 h limit), `vswarup_lab`, `standard` |
| `22_scar_array.sbatch` | scAR | 7 x (1 A30, 4 CPU, 128 GB, 6 h limit) |
| `23_ambidose_array.sbatch` | AmbiDose | 7 x (8 CPU, 64 GB, 4 h limit) |
| `30_metrics.sbatch` | scores every folder under `out/` that holds an output, one library per task | 7 x (4 CPU, 48 GB) |
| `32_collect.sbatch` | tables and the summary figure from the per-library scores | 2 CPU, 8 GB |
| `31_metrics_exp.sbatch` | scores development experiments (`out/exp_*`) apart from the benchmark tables; summarise with `summarize_experiments.py` | 7 x (4 CPU, 32 GB) |
| `submit_all.sh` | the original Glymph + CellBender + metrics submission | |

Each tool writes to `out/<tool>/`, with `<library>.time.txt` from `/usr/bin/time -v` beside its output. A folder renamed `*_failed` is ignored by the metrics job; `out/exp_*` is scored only by `31_metrics_exp.sbatch`.

Deliverables land in `out/deliverables/`: `species_mix_metrics.tsv`, `runtime.tsv`, `benchmark_summary.pdf`, and one barcode rank plot per library.

After a run is scored: `relabel_tools.py` renames tools in the run's tables by role (the CellBender 0.4.0 benchmark scores it as `cellbender040` and names it `cellbender`, keeping the tables as scored under `as_scored/`), `report_versions.py` prints the head-to-head summary, and `count_stripped_cells.py` counts, per library and tool, the cells that lost more than 10% of their counts when that is more than three times their true background plus 3 points. `05c_build_cellbender040.sbatch` builds the CellBender 0.4.0 environment; `20_cellbender_array.sbatch` and `40_scaling.sbatch` take the version to run.

Diagnostics for 0.3.2 (cells that carry expression of their own on their cluster's silent genes): `diag_heavy_cells.py` (which cells lose far more than their true background, and which genes), `diag_match.py` (the flagged cells by background match, with truth), `diag_stripped_share.py` (flagged cells that stay held to the silent genes, by share removed, against truth), `diag_profile_gap.py` (how far the background inside cells is from the empty-droplet profile, gene by gene: the feasibility check for a measured profile).

Diagnostics used while developing Glymph: `diag_budget.py` (true ambient level and swap fraction from a regression of cross-species counts on depth, beside what Glymph learned), `diag_shape.py` (does the background inside cells look like the soup or like the average cell) and `diag_exclusion.py` (can the swap fraction be read from genes that groups of cells do not express, without species labels).

## Tier 2: mouse kidney strain mixture, cells and nuclei

`config/benchmark_tier2_kidney.tsv`. GEO GSE218853 (Janssen et al. 2023): one 10x 3' v3.1 channel per library, each pooling one CAST/EiJ mouse (M. m. castaneus) with one C57BL/6J and one 129S1/SvImJ mouse (M. m. domesticus). Three libraries from cells (rep1 to rep3) and two from nuclei (nuc2, nuc3); the authors' median background per cell runs from about 2% (rep3) to about 35% (nuc2). Raw and Cell Ranger filtered matrices are on GEO per sample, 1.2 GB in all; `01_download_tier2_kidney.sh` puts them in `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix`.

Truth is the authors', kept in `data/kidney_strainmix/` (4 MB, converted to text from their repository `Hellmann-Lab/scRNA-seq_Contamination`, folder `Analysis/files`):

| File | What |
|---|---|
| `cell_metadata.tsv.gz` | 63,523 cells: replicate, barcode, cell type, strain, and `contPerCell_binom`, the background fraction per cell estimated from reads that carry the other subspecies' alleles |
| `top10_PT_markers.txt`, `panglao_PT_markers.txt` | their proximal tubule marker lists |
| `janssen_benchmark_default.tsv` | every score they published for CellBender 0.2.0, SoupX 1.6.1 and DecontX at default settings |

Metrics (`kidney_metrics.py`, tested on a synthetic library by `tests/test_kidney_metrics.py`):

| Metric | Meaning |
|---|---|
| `contam_rmsle`, `contam_kendall` | estimated fraction removed per cell against the genotype-based fraction, over every cell with a truth value; `*_cast` repeats it on castaneus cells, where the truth is measured directly |
| `removed_over_truth` | counts removed / counts the truth says are background (ideal 1) |
| `marker_removed_nonpt` | share of PT-marker counts removed from non-PT cells; PT cells are 66 to 89% of each library, so those counts are overwhelmingly background |
| `marker_kept_pt` | share of PT-marker counts kept in PT cells, where they are real |
| `slc34a1_nonpt_before`, `_after` | percent of non-PT cells with Slc34a1 detected (at least half a count) |
| `marker_log2fc` | log2 ratio of the PT-marker share of counts, PT over non-PT |
| integrity, wall time, memory | as in tier 1 |

Evaluation cells are the authors' strain-assigned singlets with a truth value. SoupX, DecontX and scAR get Cell Ranger's filtered barcodes as their cell list. AmbiDose is left out (it fails on libraries of this size).

Jobs: `submit_tier2_kidney.sh` submits the same job scripts as tier 1 with `CONFIG` and `DATA` pointing at the kidney table and folder (`--array=1-5`), then `33_kidney_metrics.sbatch` and `34_kidney_collect.sbatch` (`collect_kidney.py`: `kidney_metrics.tsv`, `runtime.tsv`, `kidney_summary.pdf`).

Count-level truth is available if wanted later: Zenodo record 15006097 holds the authors' allele count matrices (`genotyping.zip`, 450 MB) and per-library intermediates (`genotype_estimation.zip`, 1 GB).

## All tiers: 55 libraries on three platforms (prepared 2026-11-16)

One table, `config/datasets.tsv`, lists every library with its tier, platform, kind of truth and source files. Tier 1 and tier 2 are in it too, so one launcher runs everything.

| Tier | Libraries | Truth |
|---|---|---|
| 1, cell-line mixtures | 7 tuning (first run), 8 more 10x human + mouse (5' and 3'), Jurkat + 32D | species |
| 1, nuclei mixture | NEAT-seq K562 + mouse ES nuclei | species |
| 2, kidney strain mixture | 3 cell and 2 nuclei libraries (Janssen 2023) | authors' genotype-based background per cell |
| 2, xenograft | 6 mouse skin libraries with or without human immune cells | species, inside a tissue |
| 3, BD Rhapsody | human + mouse mixture x2 (from reads), demo PBMC, spleen, 5 ileum cartridges, 2 PBMC donors, liver, lung, prostate | species for the mixtures; none for the rest |
| 4, Parse Evercode | human + mouse mixture x2 sublibraries (from reads), 8 IGVF sublibraries with tissues in separate wells, bone marrow | species; tissue per well; none |
| 5, no truth | human prefrontal cortex nuclei | none |

Could not be used, and why: GSE292578 (Parse) holds counts for called cells only despite its file name; GSE337237 called cells only; the Parse-hosted and BD-hosted cell-line demos are called cells only; the pooled PBMC set with mtDNA truth (GSA-Human HRA004605) is controlled access; BD's Seven Bridges demo projects need a login; GSE147203 and GSE207393 would need 278 GB and 1.3 TB of reads realigned.

### Steps

```
02_download_manifest.sh    every file of config/datasets_manifest.tsv onto HPC3 through the transfer node (run from the Studio in tmux)
02b_wait_and_align.py      waits for the read files, restarts stalled transfers, submits alignment and ingestion
06_star_index_hgmm.sbatch  STAR index of the combined human + mouse genome
07_starsolo_species_mix.sbatch  unfiltered matrices for the two read-only mixtures (BD Rhapsody, Parse); checks md5 and which mate holds the barcodes
08_ingest_array.sbatch     ingest_dataset.py: every vendor format to one standard .h5ad (unit, pool, vendor cell calls, labels)
09_diag_prior_modes.sbatch thresholds for every library without fitting anything
submit_all_tiers.sh        prep, Glymph (default and two false-positive budgets), CellBender, SoupX, DecontX, scAR, scoring; every tool under profile.sh
13_glymph_multigpu_all.sbatch  all libraries on one 4-GPU node (throughput)
36_score_array.sbatch      species_mix_metrics.py or kidney_metrics.py by kind of truth, and generic_metrics.py for every library
```

Cell lists handed to the tools that need one (SoupX, DecontX, scAR) and used for scoring: the vendor pipeline's calls where they exist (Cell Ranger filtered matrix for kidney, BD's putative cells, STARsolo's EmptyDrops calls), otherwise the barcodes above the gap between cells and background. Glymph and CellBender get no cell list. Where CellBender's automatic priors fail on a library without a droplet plateau it is rerun with the cell count from that list (`CB_FROM_CELLS=1`), as its documentation asks, and the table says so.

Truth-free scores (`generic_metrics.py`, every library): count integrity, share removed, and exclusive-marker leakage. Exclusive markers are genes whose share of counts in one group of cells is at least 20 times their share elsewhere; groups are the tissue per well where that is a design fact, otherwise k-means clusters of the raw data, the same for every tool.

## What would count as beating CellBender on tier 1

Higher `cross_species_removed` at equal or lower `native_loss` on every dataset, zero integrity violations, better per-cell RMSLE, and wall time at least 10x lower on the same GPU type. Tier 1 alone is not enough for the claim: cell-line mixtures sit at 1 to 3% contamination, far below tissue nuclei.
