# 20261117-0544_glymph-031-dev
Request: standing directive of 2026-11-16 ("You can keep iterating to make sure that our methods come out on top and that it is robust for both 10x, BD Rhapsody as well as Parse"). Fix what the all-tier benchmark showed: thin silent-gene sets on clean libraries, libraries with no estimate, GPU memory at 80,000 cells and more.
Routing: HPC3 because Glymph needs a CUDA GPU and the data and the other tools' outputs are there.
Resources: gpu partition, one A30 per library, 4 CPUs, 48 GB, 1 h, account vswarup_lab_gpu; scoring on standard, 4 CPUs, 64 GB, account vswarup_lab.   Job/tmux: see jobs.tsv.
Inputs: 21 development libraries of config/datasets.tsv (rows 1, 9, 10, 14, 16, 17, 18, 32, 33, 44 to 55): the 12 tuning libraries, the looked-at mixtures, cortex nuclei, and four libraries opened after the benchmark (bd_hgmm_mix1, parse_hgmm_sub1, parse_hgmm_sub2, xeno_HSC_3). Kept shut for a confirmation run: seven 10x cell-line mixtures, bd_hgmm_mix2, xeno_noHSC_1 to 3, xeno_HSC_2, and every library without truth.          Env: Glymph 0.3.1.dev1 in the scanpy env (torch 2.10, CUDA 12); CellBender 0.3.2 and Glymph 0.3.0 outputs linked from 20261116-2209_glymph-alltier-bench for side-by-side scoring.
Started: 2026-11-17 05:44 EST  Finished: 2026-11-17 06:59 EST  Status: DONE
Actual: five rounds, 13 Glymph arrays of 21 libraries (1 to 4 min per library on an A30), scoring about 8 min per round; two scaling runs (80,000 cells 7 min 23 s and 25 GB RAM; 160,000 cells 12 min 48 s and 48 GB RAM).
Outputs: metrics/ (one table per library, every variant beside Glymph 0.3.0 and CellBender 0.3.2), reports_g031/ and reports_g031b/ (Glymph reports of rounds one and two), decisions.tsv (what each round changed and what it scored). Compare with code/benchmark/compare_dev_rounds.py.
Notes: the confirmation libraries were scored once with 0.3.0, so they are no longer unseen; they are not used to choose anything in this run.

## Outcome (2026-11-17)

Round three is what was frozen as 0.3.1 (tags g031c_a, g031c_b, g031e_nospare). On the 15 development mixtures, two replicates: cross-species removal 89.3% and 88.9% (0.3.0: 84.1%, CellBender 0.3.2: 83.5%), per-cell error 0.0172 and 0.0161 (0.0184, 0.0401), rank agreement 0.68 (0.59, 0.48). Kidney: error 0.089 and 0.091 (0.088, 0.097), removed over truth 1.00 (0.95, 0.96), tubule markers removed from other cells 88% (80%, 83%).
Tried and withdrawn: giving cells far above the fit their own multiplier (kidney error 0.093); sparing doublet-like cells from the exclusion constraint as the default (no gain on kidney, loss on BD Rhapsody and Parse).
These are development numbers on libraries the changes were chosen on. The test is run 20261117-0607_glymph-031-alltier.
