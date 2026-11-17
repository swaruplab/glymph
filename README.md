# Glymph

Platform-general ambient RNA removal for single-cell and single-nucleus RNA-seq, by thinning the observed counts.

The name comes from the glymphatic system, which flushes extracellular waste out of the brain. Glymph does the same for RNA that came from outside the cell.

**Status: version 0.3.1, benchmarked on all tiers, 2026-11-17.** Website: https://swaruplab.github.io/glymph/ . Slides: `reports/Glymph_overview_dark.pptx`. On 54 libraries from 10x, BD Rhapsody and Parse (`runs/20261117-0607_glymph-031-alltier/RUN.md`), against CellBender 0.3.2: about 19 times faster (median 98 s against 2122 s per library on one A30; 160,000 cells in 13 min 42 s against 4 h 29 min), lower per-cell error on 27 of 32 libraries with truth, level on removal for 10x cell-line mixtures never used in development (92.5% against 92.6% of cross-species counts), far ahead on a nuclei mixture (79% against 32%), 1.4 points behind on BD Rhapsody. It is not a clean win on low-contamination tissue: on xenograft skin it removes more cross-species counts (79% against 65%) but also about 2 times the expected same-species background. How 0.3.1 differs from 0.3.0 is in `runs/20261117-0544_glymph-031-dev/RUN.md`.

Background and the reasoning behind the design: `reports/Ambient RNA removal beyond CellBender.md`.

## The idea: measure the background, do not infer it

A generative model of a droplet cannot tell how much of a cell's RNA is background from the likelihood alone: if the model assumes too little, its decoder simply learns to "express" the background genes a little, and the fit is as good. CellBender handles this with priors and a tunable false-positive rate. Glymph measures the amount instead, from **negative controls that every library already contains**:

- A group of similar cells does not express every gene. Counts of those **silent genes** in those cells can only be background.
- Real expression lifts a gene above the background level and never below it, so across the library the silent genes are the *lowest* group of "observed over expected background", and they can be found without labels.
- Fitting each cell's silent-gene counts gives the background amounts directly: how much every partition carries, how much grows with the cell's depth, and how much is chimeric. Each cell's own silent-gene counts then say whether that cell carries more or less than its neighbours, the way reads of another genotype do in a strain mixture.

On the mouse kidney strain mixture (Janssen 2023), where the truth is known from genotypes, this recovered the structure with no labels: whole cells carry a flat amount whatever their depth, and nuclei carry a constant 0.60 empty-partition levels plus 0.44 per unit of relative depth (truth: 0.55 and 0.47).

## What it does differently from CellBender

| | CellBender | Glymph 0.3.0 |
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
3. **Calibration** (`exclusion.py`): k-means on the latent space; silent genes; a robust per-cell Poisson fit that drops cells and clusters carrying real expression (doublets); per-cell multipliers.
4. **Second fit** with the measured amounts held fixed, so the decoder gives the background back.
5. **Thinning** (`thinning.py`): integer counts, per gene, most probable entries first.

### What real data changed, in order

The diagnostics are in `code/benchmark/diag_*.py`; the dated account is in the two runs' `decisions.tsv`.

1. **The swapped fraction is not identified by the likelihood** (cell-line mixtures, 0.2.0). Read from silent genes instead.
2. **The output rule matters as much as the posterior** (0.1.0): a per-gene budget given to the most probable entries.
3. **Tissue is different from cell lines** (kidney, 0.3.0): there the background is mostly ambient, and a cluster sum cannot tell a few percent of doublets from background. The fit moved to single cells, and silent genes are taken from the lowest ridge of the whole library.
4. **Nuclei carry background that grows with their depth**, with the soup's profile (kidney nuclei, a nuclei barnyard). A third term.
5. **The mean background per cell is 1.4 to 1.6 times the median** (kidney truth). One number over-removes for the typical cell, so each cell gets its own multiplier.
6. **BD Rhapsody and split-pool data have no empty-droplet plateau.** A knee mode, and a test that a candidate plateau is really cell-free.

Known limits:
- Silent genes need at least two kinds of cells. A single homogeneous population gives no estimate; Glymph then keeps the empty-partition level and the learned swapped fraction.
- It needs background barcodes: a matrix of called cells only cannot be run (GSE292578 turned out to be one).
- The Parse path treats each round 1 well as a unit. That is by design, but only the read-based human + mouse mixture tests it against a truth.
- No pretraining, no mixed precision, no `torch.compile` yet.

## Use

```bash
export PYTHONPATH=src
python -m glymph.cli run --input raw_feature_bc_matrix.h5 --output sample.h5ad --plot sample.rank.png
python -m glymph.cli batch --manifest libs.tsv --outdir out --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Output `.h5ad`: `X` = denoised integer counts for called cells, `layers["raw"]` = the input counts, `obs` columns prefixed `glymph_` (cell probability, removed counts, background fraction, expected ambient and swapped counts), `obsm["X_glymph"]` = latent, `uns["glymph"]["report_json"]` = the run report, and a `.report.json` beside it with timings, GPU memory, learned `e` and `s`, the swap calibration and the thresholds chosen.

Inputs: 10x `.h5`, MEX folders (Cell Ranger, STARsolo, BD Rhapsody), BD Rhapsody tables (`*_MolsPerCell_Unfiltered.csv`, `*_Expression_Data_Unfiltered.st`), Parse split-pipe DGE folders, `.h5ad`. Platform structure is passed as `obs` columns of an `.h5ad` input: `--unit-key` (which barcodes share a soup) and `--pool-key` (which were amplified together). `--cell-calls` takes the vendor's called barcodes as weak labels; none are needed.

Options worth knowing: `--no-calibrate-swap` (skip calibration and the second fit), `--no-cell-multipliers` (one multiplier per pool), `--keep-cell-totals` (output rule that also keeps each cell's background total), `--fpr` (false-positive budget, default 0.004, set on the tier 1 libraries; CellBender's default is 0.01), `--s-max`.

## Tests

```bash
python tests/test_smoke.py              # unit tests plus a synthetic library, about a minute on 4 CPU threads
python tests/test_species_mix_e2e.py    # simulated barnyard through the CLI and the metrics script
python tests/test_calibration.py        # prior modes, silent-gene calibration with doublets, output rules
python tests/test_readers.py            # BD Rhapsody and Parse readers
```

The unit tests cover the integrity of every output rule, the gene budget's totals, the readers, the three prior modes, and the calibration on simulated tissue with 5% doublets (fitted background within 2 to 7% of the truth across the depth range). The simulator follows the model's own assumptions, so the synthetic library shows the code is wired correctly, not that it is accurate. Accuracy is what `code/benchmark/` measures.

## Layout

```
src/glymph/          the package (io, background, sparse, model, exclusion, thinning, api, cli, simulate)
tests/               unit and synthetic tests
code/benchmark/      download, environment builds, Slurm jobs, metrics, diagnostics (see its README)
code/dashboard/      builds the tracker file behind the project dashboard
config/              library table (datasets.tsv), download manifest, tool table, published reference values
dashboard/           the dashboard page and its data file
data/                DATA_LOCATIONS.md (large data lives on HPC3)
runs/                index.tsv, INDEX.md and one folder per run
reports/             the literature review
research_notes/      source notes behind the review
```

## License

MIT; see `LICENSE`. The icons in `code/deck/icons/` are Phosphor Icons, also MIT.
