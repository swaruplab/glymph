# 20261116-2209_glymph-alltier-bench
Request: "Keep running all the tiers ... Only once you have all the datasets ready, then you can start parallel benchmarks and do all kinds of CPU, GPU, RAM, and all sorts of benchmarks, time tests, usage tests, and everything." All tiers, every tool, profiled.
Routing: HPC3 because every tool but SoupX and DecontX needs a CUDA GPU and the data is there. Same GPU model (A30) for Glymph, CellBender and scAR so times compare.
Resources: gpu partition, one A30 per job, 4 CPUs, 64 GB (scAR 128 GB), account vswarup_lab_gpu; SoupX and DecontX on standard, 4 CPUs, 120 GB, account vswarup_lab; scoring on standard, 4 CPUs, 64 GB.   Job/tmux: see jobs.tsv.
Inputs: 54 libraries of config/datasets.tsv in the standard format (hpc3:/dfs7/swaruplab/vswarup/compute/glymph/data/std, 8 GB): 16 cell-line and 1 nuclei species mixtures (10x), 5 kidney strain-mixture libraries, 6 xenografts, 13 BD Rhapsody (2 species mixtures), 11 Parse (2 species mixtures), 1 human cortex nuclei. 12 were used to tune Glymph, 10 were looked at during development, 32 are held out.          Env: Glymph 0.3.0 in the scanpy env (torch 2.10, CUDA 12); CellBender 0.3.2 through the lab's cellbender_fix launcher, false-positive budgets 0, 0.5% and 1% from one training run; SoupX 1.6.2, DecontX (celda 1.8.0) with and without background, scAR 0.7.0. AmbiDose is left out (it failed on libraries above about 9,000 cells in the first run).
Started: 2026-11-16 22:21 EST  Finished: 2026-11-17 05:37 EST (last scaling job 05:16, rescoring 05:37)  Status: DONE
Actual: Glymph 1.9 GPU-hours for 54 libraries (median 98 s, peak 18.6 GB RAM, 21.1 GB GPU memory); CellBender 35.2 GPU-hours (median 2,122 s, peak 32.3 GB RAM); scAR 18.0 h; SoupX 2.0 h and DecontX 2.9 h on CPU. All 54 libraries on one 4-GPU node in 22 min.
Outputs: summary.tsv (means per group and tool; summary_first_pass.tsv is the version before alignment-artifact genes were masked), scores_species.tsv, scores_species_unmasked.tsv, artifact_genes.tsv, scores_kidney.tsv, scores_generic.tsv, resources.tsv, failures.tsv, one Glymph report per library (*.report.json), figures in figures/. Tool outputs stay on HPC3 (data/DATA_LOCATIONS.md).
Notes: Glymph settings were frozen before launch (decisions in runs/20261116-2010_glymph-datasets/decisions.tsv). Its false-positive budget default (0.4%) and its design were fitted on the tuning libraries; the held-out libraries are the test.

## Results (2026-11-17)

Species truth, alignment-artifact genes masked for every tool. Glymph at its default (0.4% false-positive budget), CellBender 0.3.2 at its default (1%). Removal = share of cross-species counts removed; error = per-cell RMSLE of the contamination fraction; rank = Kendall tau.

| Group (libraries) | Removal Glymph / CellBender | Error | Rank | Held out? |
|---|---|---|---|---|
| 10x cell-line mixtures (7) | 91.5% / 92.6% | 0.011 / 0.041 | 0.74 / 0.63 | yes |
| BD Rhapsody mixtures (2) | 85.3% / 92.9% | 0.015 / 0.049 | 0.62 / 0.51 | yes |
| Parse Evercode mixtures (2) | 60.5% / 62.4% | 0.029 / 0.039 | 0.33 / 0.41 | yes |
| Xenograft skin (5) | 65.4% / 67.4% | 0.030 / 0.041 | 0.27 / 0.30 | yes |
| 10x cell-line mixtures, tuning (7) | 94.9% / 95.9% | 0.014 / 0.037 | 0.70 / 0.57 | no |
| Nuclei mixture (1) | 76.0% / 32.2% | 0.032 / 0.068 | 0.62 / 0.30 | looked at |
| Kidney, genotype truth (5) | removed/truth 0.95 / 0.96 | 0.088 / 0.097 | 0.376 / 0.369 | no |

Per library: Glymph removes more than CellBender on 9 of 27 mixtures, has the lower per-cell error on 24 of 27 and the higher rank correlation on 20 of 27. With both tools at a 1% budget Glymph removes more in every group but BD Rhapsody (88.7% against 92.9%), at the price of more of the cells' own counts (for example 97.5% kept against 99.3% on the xenografts).

Where Glymph 0.3.0 falls short, and why:
- BD Rhapsody and xenografts: the gap is in the few hundred most contaminated cells (BD mix 1: 458 cells with 2 to 10% cross-species counts hold 18% of all cross-species counts; Glymph removes 57% and 24% in the two bands, CellBender 91% and 70%). These libraries are clean (40 and 29 ambient counts per empty partition), so only 18 and 5 genes per cluster pass the silent-gene test and a cell's own multiplier is shrunk to the trend.
- parse_hgmm_sub1 and hgmm_1k_v2: no silent-gene estimate at all, so the model fell back to its learned values.
- Scaling: out of GPU memory at 80,000 and 160,000 cells (4 and 8 copies of the 20,000-cell library; CellBender finished both in 1 h 46 min and 4 h 29 min with 56 and 112 GB RAM). The calibration gathered every entry of the confident cells at once.

Parse and the scoring rule: 68% of the "mouse" counts in human Parse cells are on Gm42418 and 10% on Lars2 (ribosomal RNA reads aligned to the wrong genome), present at the same level in wells that held only human cells. The scorer now masks such genes from the raw data alone (species_mix_metrics.py: at least 0.5% of one species' cross-species counts and more than ten times the gene's share in the other species' cells and in the soup). It also caught human IL6 and CSF1 in the xenograft mice (transgenes). Before the mask every tool scored about 20% on Parse.

Resources: Glymph is 19.6 times faster than CellBender on the median library (3.7 to 41 times). On 16 CPU cores without a GPU it takes 26 to 37 min, about CellBender's time on an A30. hgmm_10k_v3 takes 375 s on a V100, 216 s on an A30, 166 s on an A100. The gpu32 partition (RTX6000, L40S) refused the lab GPU account.
