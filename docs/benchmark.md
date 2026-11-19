# Benchmark

<p class="gl-kicker">54 libraries, three platforms, six tools, measured truth</p>

<div class="gl-stats">
<div class="gl-stat"><b>54</b><span>libraries</span></div>
<div class="gl-stat"><b>3</b><span>platforms</span></div>
<div class="gl-stat"><b>6</b><span>tools, same GPU model</span></div>
<div class="gl-stat"><b>12</b><span>mixtures never used in development</span></div>
</div>


Every tool ran on every library, and every score comes from a truth that does not depend on any tool:

- **27 human and mouse mixtures** (cell lines on 10x, BD Rhapsody and Parse, a nuclei mixture, six xenografts). Every count of the other species is background.
- **5 mouse kidney libraries** pooled from three strains, where genotypes give the background of every cell (Janssen et al. 2023).
- **8 Parse libraries** with one tissue per well, and **14 libraries without truth**, scored on count integrity and marker leakage.

Glymph 0.3.2 is at its default false-positive budget (0.4%), CellBender 0.4.0 at its default (1%). CellBender 0.3.2 was run on the
same libraries and scores the same as 0.4.0 within noise; both are in the tables. Glymph was run twice with the same code, and the
second run is shown beside the first, because on very clean libraries its result moves between runs. Genes that collect reads
aligned to the wrong genome, such as ribosomal RNA on mouse Gm42418, are masked for every tool by a rule that uses raw counts only.

## Speed

<div class="gl-fig" markdown>
![Wall time per library for each tool](assets/figures/dark/speed_per_library.png#only-dark)
![Wall time per library for each tool](assets/figures/light/speed_per_library.png#only-light)
</div>
<p class="gl-cap">Wall time per library on one A30 GPU (SoupX and DecontX on four CPU cores). Bars are medians over 54 libraries.</p>


Glymph's median is 98 s per library and CellBender's 36 min. The whole benchmark took
Glymph 1.9 GPU-hours and CellBender 34.5.

<div class="gl-fig" markdown>
![Wall time against library size](assets/figures/dark/scaling.png#only-dark)
![Wall time against library size](assets/figures/light/scaling.png#only-light)
</div>
<p class="gl-cap">One 20,000-cell library copied one to eight times, on one A30 with 24 GB. Time and memory only, never accuracy.</p>


At 160,000 cells Glymph takes 16 min with 46 GB of memory; CellBender takes
4 h 29 min with 128 GB.

## Per-cell accuracy

<div class="gl-fig" markdown>
![Per-cell error, Glymph against CellBender](assets/figures/dark/percell_error.png#only-dark)
![Per-cell error, Glymph against CellBender](assets/figures/light/percell_error.png#only-light)
</div>
<p class="gl-cap">One point per library with truth. Lower is better; points under the dashed line are libraries where Glymph is more accurate.</p>


Glymph has the lower per-cell error on 30 of 32 libraries with truth and the higher rank agreement on 29.

## Removal against counts kept

<div class="gl-fig" markdown>
![Removal against counts kept, by platform](assets/figures/dark/tradeoff_by_platform.png#only-dark)
![Removal against counts kept, by platform](assets/figures/light/tradeoff_by_platform.png#only-light)
</div>
<p class="gl-cap">Each line follows a tool's false-positive budget; the ring marks its default. Gold is Glymph, blue is CellBender.</p>


### Libraries never used in development

Twelve mixtures were never used to choose anything in Glymph. They are the test.

| Group | Tool | Libraries | Cross-species removed | Own counts kept | Per-cell error | Rank agreement | Adds counts |
|---|---|---|---|---|---|---|---|
| 10x cell-line mixtures | Glymph 0.3.2 | 7 | 93.8% | 98.86% | 0.008 | 0.86 | no |
| 10x cell-line mixtures | Glymph 0.3.2, second run | 7 | 93.3% | 98.86% | 0.008 | 0.86 | no |
| 10x cell-line mixtures | Glymph 0.3.1 | 7 | 92.5% | 98.93% | 0.010 | 0.78 | no |
| 10x cell-line mixtures | CellBender 0.4.0 | 7 | 92.6% | 98.72% | 0.042 | 0.63 | no |
| 10x cell-line mixtures | CellBender 0.3.2 | 7 | 92.6% | 98.74% | 0.041 | 0.63 | no |
| 10x cell-line mixtures | scAR | 7 | 96.4% | 98.69% | 0.019 | 0.38 | yes |
| 10x cell-line mixtures | SoupX | 7 | 57.5% | 99.16% | 0.024 | 0.20 | no |
| 10x cell-line mixtures | DecontX + background | 7 | 78.8% | 99.04% | 0.015 | 0.75 | no |
| 10x cell-line mixtures | DecontX | 7 | 48.7% | 99.87% | 0.023 | 0.64 | no |
| BD Rhapsody mixtures | Glymph 0.3.2 | 1 | 91.6% | 99.21% | 0.011 | 0.87 | no |
| BD Rhapsody mixtures | Glymph 0.3.2, second run | 1 | 91.5% | 99.22% | 0.011 | 0.87 | no |
| BD Rhapsody mixtures | Glymph 0.3.1 | 1 | 91.6% | 99.21% | 0.011 | 0.87 | no |
| BD Rhapsody mixtures | CellBender 0.4.0 | 1 | 92.6% | 99.56% | 0.040 | 0.64 | no |
| BD Rhapsody mixtures | CellBender 0.3.2 | 1 | 93.0% | 99.46% | 0.044 | 0.60 | no |
| BD Rhapsody mixtures | scAR | 1 | 98.9% | 97.09% | 0.029 | 0.32 | yes |
| BD Rhapsody mixtures | SoupX | 1 | 63.2% | 99.45% | 0.016 | 0.34 | no |
| BD Rhapsody mixtures | DecontX + background | 1 | 71.0% | 99.48% | 0.013 | 0.58 | no |
| BD Rhapsody mixtures | DecontX | 1 | 69.3% | 99.96% | 0.015 | 0.79 | no |
| Xenograft skin | Glymph 0.3.2 | 4 | 70.0% | 98.05% | 0.031 | 0.39 | no |
| Xenograft skin | Glymph 0.3.2, second run | 4 | 79.0% | 97.59% | 0.033 | 0.36 | no |
| Xenograft skin | Glymph 0.3.1 | 4 | 79.2% | 97.27% | 0.045 | 0.36 | no |
| Xenograft skin | CellBender 0.4.0 | 4 | 65.1% | 99.28% | 0.046 | 0.35 | no |
| Xenograft skin | CellBender 0.3.2 | 4 | 65.4% | 99.28% | 0.046 | 0.35 | no |
| Xenograft skin | scAR | 4 | 82.4% | 73.95% | 0.200 | -0.08 | yes |
| Xenograft skin | SoupX | 4 | 29.8% | 99.04% | 0.035 | -0.08 | no |
| Xenograft skin | DecontX + background | 4 | 47.9% | 96.77% | 0.079 | 0.30 | no |
| Xenograft skin | DecontX | 4 | 35.9% | 97.03% | 0.065 | 0.25 | no |


### All mixtures

| Group | Tool | Libraries | Cross-species removed | Own counts kept | Per-cell error | Rank agreement | Adds counts |
|---|---|---|---|---|---|---|---|
| 10x cell-line mixtures | Glymph 0.3.2 | 9 | 92.9% | 98.75% | 0.009 | 0.86 | no |
| 10x cell-line mixtures | Glymph 0.3.2, second run | 9 | 92.5% | 98.75% | 0.009 | 0.86 | no |
| 10x cell-line mixtures | Glymph 0.3.1 | 9 | 91.9% | 98.81% | 0.010 | 0.80 | no |
| 10x cell-line mixtures | CellBender 0.4.0 | 9 | 91.5% | 98.66% | 0.044 | 0.64 | no |
| 10x cell-line mixtures | CellBender 0.3.2 | 9 | 91.4% | 98.67% | 0.043 | 0.64 | no |
| 10x cell-line mixtures | scAR | 9 | 95.9% | 98.46% | 0.018 | 0.44 | yes |
| 10x cell-line mixtures | SoupX | 9 | 57.4% | 99.04% | 0.026 | 0.19 | no |
| 10x cell-line mixtures | DecontX + background | 9 | 80.2% | 98.89% | 0.015 | 0.78 | no |
| 10x cell-line mixtures | DecontX | 9 | 51.9% | 99.88% | 0.025 | 0.69 | no |
| 10x cell-line mixtures used for tuning | Glymph 0.3.2 | 7 | 96.0% | 98.52% | 0.013 | 0.75 | no |
| 10x cell-line mixtures used for tuning | Glymph 0.3.2, second run | 7 | 96.1% | 98.52% | 0.013 | 0.74 | no |
| 10x cell-line mixtures used for tuning | Glymph 0.3.1 | 7 | 96.1% | 98.52% | 0.013 | 0.73 | no |
| 10x cell-line mixtures used for tuning | CellBender 0.4.0 | 7 | 95.9% | 99.02% | 0.037 | 0.57 | no |
| 10x cell-line mixtures used for tuning | CellBender 0.3.2 | 7 | 95.9% | 99.02% | 0.037 | 0.57 | no |
| 10x cell-line mixtures used for tuning | scAR | 7 | 98.4% | 97.03% | 0.027 | 0.43 | yes |
| 10x cell-line mixtures used for tuning | SoupX | 7 | 72.3% | 98.69% | 0.024 | 0.17 | no |
| 10x cell-line mixtures used for tuning | DecontX + background | 7 | 77.7% | 98.61% | 0.033 | 0.70 | no |
| 10x cell-line mixtures used for tuning | DecontX | 7 | 60.1% | 99.76% | 0.038 | 0.58 | no |
| BD Rhapsody mixtures | Glymph 0.3.2 | 2 | 91.5% | 99.23% | 0.012 | 0.81 | no |
| BD Rhapsody mixtures | Glymph 0.3.2, second run | 2 | 90.1% | 99.23% | 0.012 | 0.81 | no |
| BD Rhapsody mixtures | Glymph 0.3.1 | 2 | 90.1% | 99.22% | 0.013 | 0.82 | no |
| BD Rhapsody mixtures | CellBender 0.4.0 | 2 | 92.6% | 99.43% | 0.047 | 0.53 | no |
| BD Rhapsody mixtures | CellBender 0.3.2 | 2 | 92.9% | 99.37% | 0.049 | 0.51 | no |
| BD Rhapsody mixtures | scAR | 2 | 98.2% | 96.14% | 0.038 | 0.24 | yes |
| BD Rhapsody mixtures | SoupX | 2 | 63.3% | 99.41% | 0.017 | 0.33 | no |
| BD Rhapsody mixtures | DecontX + background | 2 | 73.4% | 99.45% | 0.013 | 0.69 | no |
| BD Rhapsody mixtures | DecontX | 2 | 67.6% | 99.95% | 0.015 | 0.69 | no |
| Parse Evercode mixtures | Glymph 0.3.2 | 2 | 77.7% | 98.27% | 0.024 | 0.52 | no |
| Parse Evercode mixtures | Glymph 0.3.2, second run | 2 | 75.6% | 98.44% | 0.024 | 0.54 | no |
| Parse Evercode mixtures | Glymph 0.3.1 | 2 | 77.6% | 98.31% | 0.024 | 0.53 | no |
| Parse Evercode mixtures | CellBender 0.4.0 | 2 | 62.3% | 99.75% | 0.039 | 0.41 | no |
| Parse Evercode mixtures | CellBender 0.3.2 | 2 | 62.4% | 99.75% | 0.039 | 0.41 | no |
| Parse Evercode mixtures | scAR | 2 | 98.1% | 93.28% | 0.047 | 0.15 | yes |
| Parse Evercode mixtures | SoupX | 2 | 27.3% | 99.32% | 0.033 | 0.16 | no |
| Parse Evercode mixtures | DecontX + background | 2 | 49.0% | 97.33% | 0.036 | 0.79 | no |
| Parse Evercode mixtures | DecontX | 2 | 45.3% | 98.70% | 0.036 | 0.74 | no |
| Xenograft skin | Glymph 0.3.2 | 6 | 73.8% | 98.17% | 0.027 | 0.40 | no |
| Xenograft skin | Glymph 0.3.2, second run | 6 | 80.0% | 97.89% | 0.028 | 0.39 | no |
| Xenograft skin | Glymph 0.3.1 | 6 | 79.7% | 97.59% | 0.043 | 0.39 | no |
| Xenograft skin | CellBender 0.4.0 | 6 | 69.4% | 99.34% | 0.038 | 0.29 | no |
| Xenograft skin | CellBender 0.3.2 | 6 | 69.7% | 99.34% | 0.038 | 0.29 | no |
| Xenograft skin | scAR | 6 | 84.8% | 77.53% | 0.183 | -0.03 | yes |
| Xenograft skin | SoupX | 6 | 30.3% | 99.06% | 0.031 | -0.06 | no |
| Xenograft skin | DecontX + background | 6 | 42.8% | 97.11% | 0.072 | 0.30 | no |
| Xenograft skin | DecontX | 6 | 31.1% | 97.71% | 0.056 | 0.28 | no |
| Nuclei mixture | Glymph 0.3.2 | 1 | 79.6% | 94.94% | 0.024 | 0.75 | no |
| Nuclei mixture | Glymph 0.3.2, second run | 1 | 79.9% | 94.99% | 0.026 | 0.75 | no |
| Nuclei mixture | Glymph 0.3.1 | 1 | 79.1% | 95.07% | 0.024 | 0.75 | no |
| Nuclei mixture | CellBender 0.4.0 | 1 | 32.3% | 99.80% | 0.068 | 0.31 | no |
| Nuclei mixture | CellBender 0.3.2 | 1 | 32.2% | 99.80% | 0.068 | 0.30 | no |
| Nuclei mixture | scAR | 1 | 79.9% | 63.85% | 0.246 | -0.10 | yes |
| Nuclei mixture | SoupX | 1 | 57.6% | 94.85% | 0.034 | 0.00 | no |
| Nuclei mixture | DecontX + background | 1 | 61.8% | 95.90% | 0.035 | 0.29 | no |
| Nuclei mixture | DecontX | 1 | 13.0% | 99.92% | 0.073 | 0.58 | no |


## Tissue with genotype truth

Five mouse kidney libraries. Glymph was developed on these, so this is a check and not a test.

| Tool | Per-cell error | Rank agreement | Removed / true background | Tubule markers removed from other cell types | Tubule markers kept in tubule cells |
|---|---|---|---|---|---|
| Glymph 0.3.2 | 0.089 | 0.38 | 0.99 | 86% | 88% |
| Glymph 0.3.2, second run | 0.090 | 0.38 | 1.00 | 87% | 88% |
| Glymph 0.3.1 | 0.089 | 0.38 | 1.00 | 87% | 88% |
| CellBender 0.4.0 | 0.097 | 0.37 | 0.96 | 83% | 87% |
| CellBender 0.3.2 | 0.097 | 0.37 | 0.96 | 83% | 87% |
| scAR | 0.158 | 0.11 | 3.34 | 82% | 82% |
| SoupX | 0.143 | -0.09 | 0.40 | 22% | 97% |
| DecontX + background | 0.289 | 0.16 | 5.96 | 74% | 69% |
| DecontX | 0.205 | 0.10 | 2.80 | 70% | 73% |

## Cells that lose more than their background

<div class="gl-fig" markdown>
![Cells stripped far beyond their background, per tool](assets/figures/dark/stripped_cells.png#only-dark)
![Cells stripped far beyond their background, per tool](assets/figures/light/stripped_cells.png#only-light)
</div>
<p class="gl-cap">Cells that lost more than 10% of their counts when that is more than three times their true background plus three points, over 27 mixtures and 164,497 cells with a species truth.</p>


Glymph strips 9 cells this way, CellBender 0.4.0 4,206, scAR 9,704. Glymph 0.3.1 stripped 73:
most were the human immune cells of mouse xenografts, which sat in mouse clusters and lost nearly all their counts. Since 0.3.2 a cell
with far more counts on its cluster's silent genes than background explains is left alone, and flagged, unless those counts follow
the measured background profile and are at most a fifth of the cell.

## Where Glymph does not win yet

- **Clean tissue.** On the four xenograft libraries never used in development Glymph removes 70% of
  cross-species counts against 65% and keeps 98.0% of the cells' own counts against
  99.3%. It still removes 1.2 times the expected same-species background
  (2.0 in 0.3.1).
- **Repeat runs.** Libraries with very little background give few silent genes to test. A second run of the same code removed
  79% on those four xenografts, with a ratio of 1.7; on one of them the first run
  found no estimate and fell back to the learned values. Elsewhere the two runs differ by a median of 0.3 points.
- **BD Rhapsody removal.** 91.6% against 92.6% on the mixture never used in development,
  with a quarter of CellBender's per-cell error.
- **Own counts kept.** CellBender keeps slightly more of the cells' own counts (99.0% against 98.6%
  on the twelve libraries never seen). Part of that is same-species background it leaves in. On Parse tissue Glymph keeps
  98.0% of cluster-specific marker counts in their own cluster, CellBender 99.5%.
- **One kidney library** is cleaned 1.2 to 1.3 times beyond the genotype truth.

The full tables, one row per library and tool, are in the repository under `runs/20261117-1342_glymph-cb040-bench/`.
