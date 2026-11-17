# 20261117-0607_glymph-031-alltier
Request: standing directive of 2026-11-16 (all tiers; iterate until the method comes out on top; report honestly). Glymph 0.3.1, frozen 2026-11-17 07:03 EST, on every library, scored beside the other tools' outputs of 20261116-2209_glymph-alltier-bench.
Routing: HPC3 because Glymph needs a CUDA GPU and the data and the other tools' outputs are there.
Resources: gpu partition, one A30 per library, 4 CPUs, 48 GB, 1 h, account vswarup_lab_gpu (node hpc3-gpu-18-03 excluded: its GPU was held by another user's process); scoring on standard, 4 CPUs, 64 GB, account vswarup_lab.   Job/tmux: see jobs.tsv.
Inputs: the 54 libraries of config/datasets.tsv in the standard format (hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/std). CellBender 0.3.2 (three budgets), scAR, SoupX, DecontX and Glymph 0.3.0 are not rerun: their outputs are linked from the first all-tier run.          Env: Glymph 0.3.1 in the scanpy env (torch 2.10, CUDA 12), false-positive budgets 0.4% (default), 0, 0.2% and 1%.
Started: 2026-11-17 07:03 EST  Finished: 2026-11-17 07:49 EST (tables; the last timing jobs a few minutes later)  Status: DONE
Actual: Glymph 0.3.1 took 2.0 GPU-hours for 54 libraries at the default budget (median 98 s per library, median 3.2 GB RAM, at most 20.6 GB GPU memory); CellBender 0.3.2, from the first run, 35.2 GPU-hours (median 2122 s). 102 of 216 tasks had to be rerun because one node handed out a GPU that another process had filled.
Outputs: summary.tsv, scores_species.tsv (alignment-artifact genes masked) and scores_species_unmasked.tsv, scores_kidney.tsv, scores_generic.tsv, resources.tsv, scaling.tsv, artifact_genes.tsv, failures.tsv; first_pass/ holds the tables before the reruns. Figures: reports/deck_figures/. Tool outputs stay on HPC3.
Notes: which libraries are a clean test of 0.3.1. Never used to choose anything in 0.3.1 and scored once with 0.3.0 only: hgmm_20k_5pv2_HT, hgmm_10k_5p_gemx, hgmm_10k_3p_chromX, hgmm_5k_3p_multiplex, hgmm_1k_3p_LT, hgmm_10k_v3_DI, hgmm_1k_v2, xeno_noHSC_1, xeno_noHSC_2, xeno_noHSC_3, xeno_HSC_2, bd_hgmm_mix2 (12 libraries, "confirmation"). Opened for 0.3.1 development after the first benchmark: bd_hgmm_mix1, parse_hgmm_sub1, parse_hgmm_sub2, xeno_HSC_3. The species score masks alignment-artifact genes, a rule added after the first benchmark for every tool.

## Results (2026-11-17)

Glymph 0.3.1 at its default budget (0.4%) against CellBender 0.3.2 at its default (1%). Glymph 0.3.0 is in the tables as glymph_030.

| Group (libraries) | Removal, 0.3.1 / CellBender | Own counts kept | Per-cell error | Rank agreement |
|---|---|---|---|---|
| Never used in development: 10x cell lines (7) | 92.5% / 92.6% | 98.93% / 98.74% | 0.010 / 0.042 | 0.78 / 0.63 |
| Never used: BD Rhapsody (1) | 91.6% / 93.0% | 99.21% / 99.46% | 0.011 / 0.044 | 0.87 / 0.60 |
| Never used: xenograft skin (4) | 79.2% / 65.4% | 97.27% / 99.28% | 0.045 / 0.046 | 0.36 / 0.35 |
| All 27 mixtures | 88.6% / 83.5% | 98.32% / 99.08% | 0.020 / 0.042 | 0.67 / 0.50 |
| Kidney, genotype truth (5, tuned on) | removed/truth 1.00 / 0.96 | tubule markers removed elsewhere 87% / 83% | 0.089 / 0.097 | 0.38 / 0.37 |

Lower per-cell error than CellBender on 27 of 32 libraries with truth, higher rank agreement on 28. Nuclei mixture: 79.1% removed against 32.2%. Parse mixtures (opened for development): 77.6% against 62.4%.
Scaling on one A30 (20,000 to 160,000 cells): 3 min 26 s, 4 min 56 s, 7 min 31 s, 13 min 42 s; 47.9 GB RAM and 18.6 GB GPU memory at the largest (CellBender: 4 h 29 min, 112 GB).

Where 0.3.1 does not win: on xenograft skin it removes 2.0 times the expected same-species background (native loss 0.4 to 2.1%) and its per-cell error is no better than CellBender's; on BD Rhapsody it is 1.4 points behind on removal; on Parse tissue libraries it keeps 93.7% of cluster-specific marker counts in their own cluster against 99.5%; libraries with few testable silent genes vary between runs.
