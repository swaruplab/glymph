---
hide:
  - navigation
  - toc
---

<p class="gl-kicker">Swarup Lab · UCI MIND</p>

# Glymph

<p class="gl-lede">Ambient RNA removal for single-cell and single-nucleus RNA-seq that measures the background instead of inferring it.</p>

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[The benchmark](benchmark.md){ .md-button }
[Slides](https://github.com/swaruplab/glymph/blob/main/reports/Glymph_overview_dark.pdf){ .md-button }
[Code on GitHub](https://github.com/swaruplab/glymph){ .md-button }

<div class="gl-stats">
<div class="gl-stat"><b>19×</b><span>faster per library than CellBender, on the same GPU</span></div>
<div class="gl-stat"><b>27 of 32</b><span>libraries with truth where Glymph is more accurate per cell</span></div>
<div class="gl-stat"><b>3</b><span>platforms: 10x Chromium, BD Rhapsody, Parse Evercode</span></div>
<div class="gl-stat"><b>0</b><span>counts added: the output is whole numbers, never above the raw count</span></div>
</div>


## The idea in one paragraph

Every droplet, well or split-pool barcode carries RNA its cell did not make. Existing deep-learning removers infer how much from a
likelihood and priors, and the likelihood cannot pin that amount down. Glymph measures it. A gene that a cell type does not express
is a **silent gene** there: it should read zero, so every count it shows is background. Each library contains thousands of these
internal negative controls. Glymph reads the background off them, per library and per cell, holds it fixed, and lets a small
autoencoder separate expression from background.

<div class="gl-cards">
<div class="gl-card"><h3>Measured, not inferred</h3><p>The amount of background comes from silent genes, checked against genotype truth in mouse kidney.</p></div>
<div class="gl-card"><h3>Fast</h3><p>A median of 98 s per library on one GPU, against 35 min for CellBender. 160,000 cells in 14 min.</p></div>
<div class="gl-card"><h3>Three platforms</h3><p>Droplets, microwells and split-pool barcoding, with no vendor cell calls needed.</p></div>
<div class="gl-card"><h3>Whole counts</h3><p>Integers in, integers out. No entry is ever higher than it was.</p></div>
</div>


<div class="gl-fig" markdown>
![Per-cell error of Glymph against CellBender, one point per library](assets/figures/dark/percell_error.png#only-dark)
![Per-cell error of Glymph against CellBender, one point per library](assets/figures/light/percell_error.png#only-light)
</div>
<p class="gl-cap">Per-cell error of the estimated background fraction, one point per library with truth. Points under the dashed line are libraries where Glymph is more accurate.</p>


## Install

```bash
git clone https://github.com/swaruplab/glymph.git
cd glymph
pip install -e .
glymph run --input raw_feature_bc_matrix.h5 --output sample.h5ad
```

Glymph needs PyTorch and runs on one CUDA GPU; it also runs on CPU. See the [quickstart](quickstart.md).

<p class="gl-take"><b>TAKE-HOME</b>Measure the background. Do not infer it.</p>

