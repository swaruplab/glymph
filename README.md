# Glymph

Platform-general ambient RNA removal for single-cell and single-nucleus RNA-seq, by thinning the observed counts.

The name comes from the glymphatic system, which flushes extracellular waste out of the brain. Glymph does the same for RNA that came from outside the cell.

**Status: version 0.4.0rc2, a release candidate, confirmed on 54 libraries beside CellBender 0.4.0 on 2026-11-19** (`runs/20261118-2355_glymph-040rc2-bench/RUN.md`). On the 12 human + mouse mixtures never used in development it removes 88.6% of the cross-species counts against 83.4% (0.3.2: 85.7%), with a per-cell error of 0.019 against 0.047, and two runs agree within 0.12 points. Cross-species removal by platform, all mixtures of each kind: 10x cell lines 93.7% against 91.5%, xenografts 81.4% against 69.4%, BD Rhapsody 93.0% against 92.6%, Parse 80.2% against 62.3%, a nuclei mixture 79.8% against 32.3%. On mouse kidney with genotype truth the two tools are level (per-cell error 0.090 against 0.097). Median 98 s against 2,135 s per library on one A30. It is a candidate, not a release: its open faults are listed under Known limits. The website and the slides have not been rebuilt for 0.4.0: the site (https://swaruplab.github.io/glymph/) shows 0.3.1, and `docs/` and `reports/Glymph_overview_dark.pptx` describe 0.3.2 against CellBender 0.4.0 (`runs/20261117-1342_glymph-cb040-bench/RUN.md`).

Background and the reasoning behind the design: `reports/Ambient RNA removal beyond CellBender.md`.

## The idea: measure the background, do not infer it

A generative model of a droplet cannot tell how much of a cell's RNA is background from the likelihood alone: if the model assumes too little, its decoder simply learns to "express" the background genes a little, and the fit is as good. CellBender handles this with priors and a tunable false-positive rate. Glymph measures the amount instead, from **negative controls that every library already contains**:

- A group of similar cells does not express every gene. Counts of those **silent genes** in those cells can only be background.
- Real expression lifts a gene above the background level and never below it, so across the library the silent genes are the *lowest* group of "observed over expected background", and they can be found without labels.
- Fitting each cell's silent-gene counts gives the background amounts directly: how much every partition carries, how much grows with the cell's depth, and how much is chimeric. Each cell's own silent-gene counts then say whether that cell carries more or less than its neighbours, the way reads of another genotype do in a strain mixture.

On the mouse kidney strain mixture (Janssen 2023), where the truth is known from genotypes, this recovered the structure with no labels: whole cells carry a flat amount whatever their depth, and nuclei carry a constant 0.60 empty-partition levels plus 0.44 per unit of relative depth (truth: 0.55 and 0.47).

## What it does differently from CellBender

| | CellBender | Glymph |
|---|---|---|
| How much background | inferred by the likelihood, held in place by priors | measured from silent genes (`exclusion.silent_budget`), per library and per cell (`cell_multipliers`), then held fixed |
| Kinds of background | ambient, plus swapping in the full model | ambient per partition; ambient that grows with depth (nuclei); chimeric counts with the pool's average cell profile |
| Background structure | one ambient profile per run | one soup per **unit**, one chimera profile per **pool** (10x: library and library; Parse: round 1 well and sublibrary; Rhapsody: cartridge) |
| Where cells end | needs an empty-droplet plateau, or expected cells and total droplets from the user | plateau when there is one (checked to be cell-free by its gene-count dispersion), otherwise the knee of the rank curve; no vendor cell calls needed. On BD Rhapsody the knee gives 8,061 cells where the vendor calls 8,061, and within about 10% on five more cartridges |
| Training | Pyro SVI, 150 epochs, batch 128, dense likelihood | plain PyTorch, a few thousand steps at batch 1,024, likelihood on nonzero entries only, library held as CSR on the GPU |
| Integer output | posterior sampling plus a knapsack solver | same objective in closed form (water-filling and a bisection); optionally also keeps each cell's background total (`thinning.gene_cell_budget`) |
| Count integrity | subtraction only | subtraction only, by construction (`0 <= output <= raw`) |
| Multi-GPU | none | one library per GPU (`glymph batch`, or a Slurm array) |

## Model

For barcode `n` with counts `x[n, :]` summing to `N[n]`, unit `u`, pool `p`:

```
empty partition (y = 0)
  x | N ~ Multinomial(N, (1 - s_p) * soup_u + s_p * chibar_p)
  N     ~ NegBinomial(mean = e_u / (1 - s_p), shape from the spread of the empty partitions)

cell (y = 1)
  ambient UMIs   eps[n] * e_u                     profile soup_u     per partition; eps[n] near k[n]
  swapped UMIs   s_p * N[n]                       profile chibar_p   proportional to depth
  native UMIs    d[n] = N[n](1 - s_p) - eps[n]e_u profile chi[n] = decoder(z[n])
  x | N ~ Multinomial(N, mixture of the three)

measured, not learned
  k[n] = kappa + kappa_d * N[n] / N_ref, then adjusted by the cell's own silent-gene counts
  s_p, kappa, kappa_d   from the silent-gene fit

output
  background share of each nonzero entry  p = (ambient + swapped rate) / total rate
  per gene: remove round(sum over cells of x * p) counts, from the entries where p is highest
```

`soup_u`, `chibar_p` and `e_u` are computed before training (`background.py`). `z`, `eps` and `y` come from an MLP encoder; `y` is enumerated exactly.

### Steps

1. **Priors** (`background.estimate_priors`): empty-partition level, soup profile, cell threshold. Three modes, chosen from the data: plateau, knee, order of magnitude. Also a confident-cell depth (the lowest gap above the threshold).
2. **First fit**: the model learns cell profiles and a latent space.
3. **Calibration** (`exclusion.py`): k-means on principal components of the counts (`multires.pca_embedding`), so that two runs give the same clusters; silent genes; a robust per-cell Poisson fit that drops cells and clusters carrying real expression (doublets); per-cell multipliers.
4. **Second fit** with the measured amounts held fixed, so the decoder gives the background back.
5. **Thinning** (`thinning.py`): integer counts, per gene, most probable entries first.

### What real data changed, in order

The diagnostics are in `code/benchmark/diag_*.py`; the dated account is in each run's `decisions.tsv` (for 0.4.0: `runs/20261118-1245_glymph-040-dev/decisions.tsv`).

1. **The swapped fraction is not identified by the likelihood** (cell-line mixtures, 0.2.0). Read from silent genes instead.
2. **The output rule matters as much as the posterior** (0.1.0): a per-gene budget given to the most probable entries.
3. **Tissue is different from cell lines** (kidney, 0.3.0): there the background is mostly ambient, and a cluster sum cannot tell a few percent of doublets from background. The fit moved to single cells, and silent genes are taken from the lowest ridge of the whole library.
4. **Nuclei carry background that grows with their depth**, with the soup's profile (kidney nuclei, a nuclei barnyard). A third term.
5. **The mean background per cell is 1.4 to 1.6 times the median** (kidney truth). One number over-removes for the typical cell, so each cell gets its own multiplier.
6. **BD Rhapsody and split-pool data have no empty-droplet plateau.** A knee mode, and a test that a candidate plateau is really cell-free.
7. **A cell placed in the wrong cluster must not be held to that cluster's silent genes** (xenograft skin, 0.3.2). The few human immune cells of a mouse xenograft sat in mouse clusters, every human gene was silent there, and 0.3.1 removed 92 to 99% of their counts. A cell with far more counts on its cluster's silent genes than background explains is now cleaned that way only when those counts follow the measured background profile (`exclusion.background_match`) and are at most a fifth of the cell. Otherwise it carries expression of its own there (a rare cell type, or a doublet): it is left alone, cleaned at the ordinary level, and flagged (`glymph_doublet_like`). The species and genotype truths set the two limits: below them the flagged cells carry at least as much background as is removed, above them two to twenty times less.
8. **The calibration clusters must be the same in every run** (0.4.0). Clusters built on the model's latent space differ between two runs on a GPU, and on libraries with few testable genes the background level moved with them (1 to 2 points of removal on average, up to 17). Clusters are now built on principal components of the counts; two runs agree within about 0.1 point.
9. **Depth-proportional background has two possible profiles, and a sum cannot tell them apart** (0.4.0). Summed over a cell's silent genes, ambient RNA that grows with depth and swapped molecules look alike, and the fit left the swapped fraction at zero: 14 to 28% too little background on three cell-line mixtures. Gene by gene they differ (cells hold more nuclear and mitochondrial RNA than the soup does), so the split is searched: the proportion at which the most candidate genes line up on one level wins (`exclusion._swap_shape`). It is used only where the depth-proportional background is at most 5% of depth; on kidney nuclei the same search over-cleaned by the genotype truth, because related cell types express each other's genes weakly and in the proportions of the average cell.
10. **What is right on cell-line mixtures can be wrong in tissue** (0.4.0). Three additions gained 2 to 4 points on mixtures, where every cross-species count is background by construction, and failed on kidney with genotype truth or on tissue markers: a first multi-resolution tree that set sub-clusters aside, a per-gene background level shared across clusters (`multires.shared_level`), and a rule that gives a cluster of dirty cells its neighbour's silent genes (`superpose`). All three are in the code, off by default, with the reason in their docstrings. The tree that is on can only protect: a sub-cluster overrules its cluster for a gene that a quarter of its cells carry. Spliced against unspliced molecules for nuclei (`splice.py`, `--splice`, needs `layers["unspliced"]`) is a real signal on average and too weak per gene and cluster to change the kidney scores; it is opt-in.
11. **A background fit can run away** (0.4.0). The pairs of genes and clusters tested for the level can follow the fit or be settled at the start; under either rule a larger background can make weakly expressed genes count as silent, which pulls the level up further (one BD Rhapsody PBMC library ended at 17% of depth where 3% is right). Both rules are now run, and where they differ by more than half the lower answer is kept (`calib_cross_check`): expression can only add counts to genes taken for silent.
12. **The benchmark's truth for same-species background needed a second version** (0.4.0). The usual extrapolation multiplies a cell's cross-species counts by the species split of the empty partitions. Measured where it is visible (human RNA in mouse cells against mouse RNA in human cells), the background inside cells has a different split on several libraries (`code/benchmark/diag_species_split.py`). `species_mix_metrics.py` reports both; columns ending in `_measured` use the measured split.

Known limits:
- Silent genes need at least two kinds of cells. A single homogeneous population gives no estimate; Glymph then keeps the empty-partition level and the learned swapped fraction.
- It needs background barcodes: a matrix of called cells only cannot be run (GSE292578 turned out to be one).
- The Parse path treats each round 1 well as a unit. That is by design, but only the read-based human + mouse mixture tests it against a truth.
- Libraries with very little background give few silent genes to test, and the estimate then varies between runs of the same code; on the cleanest ones a run may find no estimate at all and fall back to the learned values, which removes less.
- Open faults of 0.4.0rc2, found by its confirmation run: one clean xenograft library gets no silent-gene estimate (64% of cross-species counts removed against CellBender's 78%); on another, 140 of 3,312 cells lose far more than their true background (over all 27 mixtures 144 cells of 164,497, against 9 for 0.3.2 and 4,206 for CellBender 0.4.0); the sublibraries of a Parse bone marrow experiment disagree about sixfold on the background level; about 150 cells of a BD Rhapsody cell-line mixture that carry 5 to 16% of the other species keep it.
- On kidney tissue with genotype truth Glymph is level with CellBender 0.4.0, not ahead of it by the margin seen on mixtures.
- No pretraining, no mixed precision, no `torch.compile` yet.

## Use

```bash
export PYTHONPATH=src
python -m glymph.cli run --input raw_feature_bc_matrix.h5 --output sample.h5ad --plot sample.rank.png
python -m glymph.cli batch --manifest libs.tsv --outdir out --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Output `.h5ad`: `X` = denoised integer counts for called cells, `layers["raw"]` = the input counts, `obs` columns prefixed `glymph_` (cell probability, removed counts, background fraction, expected ambient and swapped counts; `glymph_doublet_like` for cells left alone because they carry expression of their own on their cluster's silent genes, with `glymph_background_match` and `glymph_silent_share` for every cell that was examined), `obsm["X_glymph"]` = latent, `uns["glymph"]["report_json"]` = the run report, and a `.report.json` beside it with timings, GPU memory, learned `e` and `s`, the swap calibration and the thresholds chosen.

Inputs: 10x `.h5`, MEX folders (Cell Ranger, STARsolo, BD Rhapsody), BD Rhapsody tables (`*_MolsPerCell_Unfiltered.csv`, `*_Expression_Data_Unfiltered.st`), Parse split-pipe DGE folders, `.h5ad`. Platform structure is passed as `obs` columns of an `.h5ad` input: `--unit-key` (which barcodes share a soup) and `--pool-key` (which were amplified together). `--cell-calls` takes the vendor's called barcodes as weak labels; none are needed.

Options worth knowing: `--no-calibrate-swap` (skip calibration and the second fit), `--no-cell-multipliers` (one multiplier per pool), `--keep-cell-totals` (output rule that also keeps each cell's background total), `--fpr` (false-positive budget, default 0.003, the value at which same-species removal equals its expected amount on the development mixtures; CellBender's default is 0.01), `--s-max`, `--doublet-match` and `--doublet-share-max` (when a cell with a large excess on silent genes is left alone: profile match below 0.5, or more than 20% of its counts there; `--spare-doublets` leaves every such cell alone), `--calib-embedding latent` (calibration clusters on the latent space, as up to 0.3.2), `--no-swap-contrast`, `--no-calib-cross-check`, `--splice` (use `layers["unspliced"]` of an `.h5ad` input, nuclei).

## Tests

```bash
python tests/test_smoke.py              # unit tests plus a synthetic library, about a minute on 4 CPU threads
python tests/test_species_mix_e2e.py    # simulated barnyard through the CLI and the metrics script
python tests/test_calibration.py        # prior modes, silent-gene calibration with doublets, output rules
python tests/test_readers.py            # BD Rhapsody and Parse readers
python tests/test_large_path.py         # block-wise paths for large libraries give the single-pass answer
python tests/test_background_match.py   # the profile test that tells a contaminated cell from one with a second expression program
python tests/test_multires.py           # reproducible clusters, sub-clusters, the level shared across clusters
python tests/test_splice.py             # spliced against unspliced molecules as a second negative control
python tests/test_superpose.py          # a cluster of dirty cells against a true cell type
```

The unit tests cover the integrity of every output rule, the gene budget's totals, the readers, the three prior modes, and the calibration on simulated tissue with 5% doublets (fitted background within 2 to 7% of the truth across the depth range). The simulator follows the model's own assumptions, so the synthetic library shows the code is wired correctly, not that it is accurate. Accuracy is what `code/benchmark/` measures.

## Layout

```
src/glymph/          the package (io, background, sparse, model, exclusion, multires, splice, thinning, api, cli, simulate)
tests/               unit and synthetic tests
code/benchmark/      download, environment builds, Slurm jobs, metrics, diagnostics (see its README)
code/dashboard/      builds the tracker file behind the project dashboard
code/deck/           builds the overview deck and the figures the website shares with it
code/site/           writes the website's pages from the benchmark tables
code/hpc_relay/      reaching HPC3 through an on-campus machine when off the VPN
config/              library table (datasets.tsv), download manifest, tool table, published reference values
dashboard/           the dashboard page and its data file
data/                DATA_LOCATIONS.md (large data lives on HPC3)
runs/                index.tsv, INDEX.md and one folder per run
reports/             the literature review
research_notes/      source notes behind the review
```

## License

MIT; see `LICENSE`. The icons in `code/deck/icons/` are Phosphor Icons, also MIT.
