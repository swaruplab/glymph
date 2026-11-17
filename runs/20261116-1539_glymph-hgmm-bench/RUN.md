# 20261116-1539_glymph-hgmm-bench

Request: launch the tier 1 benchmark (Glymph against CellBender on seven 10x human + mouse mixtures), run everything in parallel, install the packages for the other baselines, document everything, keep a project dashboard. Vivek was away and asked for all decisions to be made independently.

Routing: HPC3, because CUDA GPUs are needed for a like-for-like speed comparison with CellBender. Inputs are 1.2 GB and were fetched straight onto HPC3 through `hpc3-xfer`.

Resources (every GPU job asks for A30s so run times compare):

| Job | Account, partition | Per task |
|---|---|---|
| Glymph, one library per GPU | `vswarup_lab_gpu`, `gpu` | 1 A30, 4 CPU, 48 GB, 1 h limit |
| Glymph, seven libraries on one node | `vswarup_lab_gpu`, `gpu` | 4 A30, 16 CPU, 128 GB, 1 h limit |
| CellBender 0.3.2 | `vswarup_lab_gpu`, `gpu` | 1 A30, 4 CPU, 64 GB, 8 h limit |
| scAR 0.7.0 | `vswarup_lab_gpu`, `gpu` | 1 A30, 4 CPU, 128 GB, 6 h limit |
| SoupX, DecontX | `vswarup_lab`, `standard` | 4 CPU, 64 GB, 6 h limit |
| AmbiDose 0.5.15 | `vswarup_lab`, `standard` | 8 CPU, 64 GB, 4 h limit |
| Input prep, metrics, environment builds | `vswarup_lab`, `standard` | 2 to 4 CPU, 24 to 48 GB |

Job/tmux: 53 Slurm jobs, with one note each, in `jobs.tsv` in this folder. No tmux session.

Run directory: `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/20261116-1539_glymph-hgmm-bench`

Inputs: seven raw 10x matrices listed in `config/benchmark_tier1.tsv`, at `hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm` (1.2 GB).

Env:
- Glymph: existing `scanpy` environment (Python 3.12, torch 2.10, CUDA 12), code loaded from the run folder with `PYTHONPATH`. Nothing installed into it.
- CellBender 0.3.2: the lab's `/dfs7/swaruplab/vswarup/bin/cellbender_fix` launcher and venv (torch 2.0.1+cu118, pyro-ppl 1.8.6).
- New environments under `/dfs7/swaruplab/vswarup/bin/miniconda_envs/`, built inside compute jobs:
  - `glymph_bench_r` (1.0 GB): R 4.5.3, SoupX 1.6.2, decontX 1.8.0, Seurat 5.5.1, rhdf5 2.56.1
  - `glymph_bench_scar` (2.4 GB): Python 3.10, scAR 0.7.0 from github.com/Novartis/scar, torch 2.14.1
  - `glymph_bench_ambidose` (0.3 GB): Python 3.11, AmbiDose 0.5.15 from PyPI

Started: 2026-11-16 15:39 EST (MacBook clock)  Finished: 2026-11-16 18:34 EST  Status: DONE

Actual: elapsed 2 h 55 min for 53 Slurm jobs, including every rerun and the development experiments. Peak memory per task: scAR 21 GB, CellBender 16 GB, SoupX and DecontX 10 GB, Glymph 8 GB RAM and 5 GB GPU. GPU time: CellBender 4.6 h in total, scAR 1.5 h, all Glymph runs together under 2 h.

Outputs in this folder: `species_mix_metrics.tsv`, `runtime.tsv`, `benchmark_summary.pdf`, one barcode rank plot per library, environment version files, `jobs.tsv`, `decisions.tsv`. Large outputs stay on HPC3 and are listed in `data/DATA_LOCATIONS.md`.

Dashboard: https://claude.ai/artifact/N5U66VDjSQLW4pMZPoEBqT (private until shared). It reads `dashboard/tracker.json`, rebuilt by `code/dashboard/build_tracker.py`.

## Results

Means over the seven libraries, from `species_mix_metrics.tsv` and `runtime.tsv` in this folder. The figure is `benchmark_summary.pdf`.

| Tool | Cross-species removed, % | Same-species kept, % | Native loss, % | Per-cell error | Entries above raw | Fractional entries | Wall per library, s |
|---|---|---|---|---|---|---|---|
| Glymph 0.2.0 (default, budget 0.5%) | 93.3 | 98.97 | 0.005 | 0.0103 | 0 | 0 | 184 |
| Glymph 0.2.0, budget 0 | 90.7 | 99.31 | 0.000 | 0.0124 | 0 | 0 | 138 |
| Glymph 0.2.0, budget 1% | 94.1 | 98.60 | 0.121 | 0.0097 | 0 | 0 | 141 |
| CellBender 0.3.2 | 94.3 | 99.20 | 0.024 | 0.0167 | 0 | 0 | 2344 |
| scAR 0.7.0 | 96.8 | 97.79 | 0.829 | 0.0167 | 167 million | 0 | 780 |
| AmbiDose 0.5.15 (3 of 7 libraries) | 85.7 | 99.69 | 0.000 | 0.0191 | 0 | 0 | 198 |
| SoupX 1.6.2 | 69.7 | 98.75 | 0.050 | 0.0131 | 0 | 291 million | 160 |
| DecontX + background | 72.1 | 98.77 | 0.000 | 0.0203 | 0 | 294 million | 217 |
| DecontX 1.8.0 | 54.6 | 99.83 | 0.000 | 0.0252 | 0 | 292 million | 304 |
| Glymph 0.1.0 | 81.8 | 99.74 | 0.000 | 0.0168 | 0 | 0 | 90 |
| Glymph 0.0.1 | 72.0 | 99.54 | 0.000 | 0.0164 | 0 | 0 | 88 |

"Budget" is the false-positive budget: extra counts a tool may remove beyond the background it expects, as a share of each gene's native counts. CellBender's default is 1%. Glymph's default of 0.5% was chosen on this benchmark, so the other two settings are shown beside it.

Glymph 0.2.0 against CellBender 0.3.2, library by library (cross-species counts removed, %):

| Library | Cross-species in raw, % | Glymph 0.2.0 | CellBender 0.3.2 | Glymph wall, s | CellBender wall, s |
|---|---|---|---|---|---|
| hgmm_10k_gemx_v4 | 1.07 | 96.1 | 96.6 | 202 | 4179 |
| hgmm_10k_v3 | 1.49 | 93.6 | 95.7 | 220 | 2921 |
| hgmm_12k_v2 | 1.70 | 97.9 | 97.5 | 167 | 2858 |
| hgmm_1k_v3 | 0.32 | 82.2 | 86.6 | 142 | 465 |
| hgmm_20k_v31_HT | 1.32 | 92.1 | 94.7 | 227 | 3568 |
| hgmm_5k_v3 | 1.02 | 94.5 | 94.5 | 164 | 1034 |
| hgmm_6k_v2 | 1.44 | 96.4 | 94.5 | 165 | 1384 |

Multi-GPU: `glymph batch` did all seven libraries in 269 s wall on one node with four A30s. The seven single-GPU Glymph runs sum to 1,286 s and the seven CellBender runs to 16,409 s.

What this shows and what it does not:
- Glymph 0.2.0 is about level with CellBender on these mixtures, not ahead of it. It removes one point less background on average, more on two libraries, the same on one and less on four. At equal retention (compare "budget 0", which keeps 99.31%, with CellBender's 99.20%) CellBender removes about 3.5 points more.
- Glymph has the lower native loss at its default, the lower per-cell contamination error at every setting, never adds a count, and takes about a thirteenth of the wall time per library. Timings were taken on shared nodes that slowed during the afternoon; Glymph 0.1.0, run earlier on quieter nodes, took 30 to 155 s.
- scAR removes the most but writes 167 million entries above the raw counts across the seven libraries and loses the most native counts. SoupX and DecontX remove 55 to 72% and return fractional counts.
- AmbiDose is scored on three libraries only; it fails inside scanpy on the four larger ones.
- Cell-line mixtures carry 0.3 to 1.7% cross-species counts. Tissue nuclei carry far more, and that is where the exclusion bound and the claim of matching CellBender both still have to be tested.

## What happened, in order

The full dated log is `decisions.tsv`. The short version:

1. First contact with a GPU exposed two Glymph bugs, both fixed within minutes: a CUDA statistics call before the CUDA context existed, and `glymph batch` exiting 0 when every library had failed.
2. CellBender 0.3.2 trained and then died at its checkpoint save. This is the fault the lab documented the same day in `cellbender_fix`; the benchmark now runs CellBender through that launcher. Training and inference are unchanged.
3. Baselines added: SoupX, DecontX (with and without empty droplets as background), scAR and AmbiDose. CellSweep was left out because no public package could be found. The PyPI project named `scar` is an unrelated tool; scAR has to come from GitHub.
4. Glymph v0 (0.0.1) scored 63 to 79% cross-species removal with zero native loss and no integrity violations, in 33 to 148 s per library. Fast and safe, but well short of CellBender.
5. Diagnosis (`diag_budget.py`): Glymph's ambient level was right, its swap fraction was 2 to 2.5 times too low. A prior pushing the swap fraction up made it worse, because the gene profile of the background inside cells differs from both fixed profiles and the network explains it better (`diag_shape.py`). Longer training, no early stopping and smaller batches changed nothing. Fixing the swap fraction near its true value gave 93 to 97%.
6. Glymph 0.1.0 changed the output rule to a per-gene budget given to the most probable entries: 76 to 87%.
7. Glymph 0.2.0 reads the swap fraction from the data without labels (the exclusion bound, `diag_exclusion.py` then `src/glymph/exclusion.py`), retrains with it fixed, and adds an optional false-positive budget: the results above.

## Notes

- Evaluation cells are the barcodes at or above the order-of-magnitude threshold in the raw matrix with at least 90% of counts from one species. SoupX, DecontX, scAR and AmbiDose are given every barcode at or above that threshold as their cell list, so no tool is scored on barcodes it was not given.
- The two Cell Ranger v2 libraries (6k, 12k) hold one matrix per genome. `prep_inputs.py` rewrites every library into one v3-layout file for the baselines; CellBender gets that file for the two v2 libraries and the original file for the rest.
- One of four allocated A30s was already holding 19.75 GB from another process during the second 4-GPU attempt. `glymph batch` now stops feeding a device that runs out of memory and retries on the others.
- Threshold observation for the routing rules: nothing here came close to a limit. Peak RAM was 8 GB for Glymph and about 21 GB for scAR; `/dfs7` stayed at 86%.
- GPU nodes and `/dfs7` slowed about five-fold for part of the afternoon (same code, same steps). Scoring all tools in one job then took 30 minutes; split into one task per library it takes two.
- Nothing was deleted on HPC3. The run folder holds 44 GB, of which about 25 GB is experiments, failed attempts, the shared inputs and the first 4-GPU run. See `data/DATA_LOCATIONS.md`.
- Timestamps in `decisions.tsv` were corrected at the end from the scheduler's submit and end times; a few are accurate only to within a few minutes.
