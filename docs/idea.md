# The idea

<p class="gl-kicker">The gap, and what Glymph does about it</p>

## Every count has three possible sources

A count in the matrix can come from the cell in the partition, from free RNA in the suspension that every partition captures
(**ambient RNA**), or from a molecule that changed barcode during amplification (**swapped reads**). Only the first is biology.
Background is not a rare artifact: published estimates are 3 to 35% of a cell's counts in mouse kidney, with medians of 17% and
35% in nuclei from frozen tissue (Janssen et al. 2023, Genome Biology).

## What was missing

<div class="gl-cards">
<div class="gl-card"><h3>Speed</h3><p>The standard tool trains a new model on every library. In our benchmark its median was 36 min per library on one A30 GPU, and 4 h 29 min for 160,000 cells.</p></div>
<div class="gl-card"><h3>A measured amount</h3><p>A flexible model can explain background as expression and fit the data just as well, so priors decide the amount. CellBender removed 74% of the expected same-species background across 27 mixtures.</p></div>
<div class="gl-card"><h3>More than one platform</h3><p>One plateau of empty droplets and one ambient profile per run is a 10x premise. BD Rhapsody and Parse Evercode fall outside it.</p></div>
<div class="gl-card"><h3>Intact counts</h3><p>The faster alternatives return fractional counts or, in one case, counts higher than the raw data.</p></div>
</div>


## Silent genes are built-in negative controls

A gene that a cell type does not express should read zero in those cells. Whatever it does read is background. This is the same
logic as reads of another species in a mixture, or of another genotype in a pooled sample, but it needs no special experiment.

Glymph clusters the cells on the model's own embedding, finds for every cluster the genes whose counts are no more than background
alone explains, and uses the counts on those genes as the measurement. No labels and no marker lists are involved.

## Three amounts, each read off the data

| Amount | What it is |
|---|---|
| Ambient, per partition | A constant amount, in units of what an empty partition holds |
| Ambient, with depth | Ambient RNA that grows with the depth of the cell; nuclei carry it |
| Swapped reads | Proportional to depth, with the average profile of the pool |

In mouse kidney nuclei pooled from three strains, genotypes give the true background of every nucleus: a constant of 0.55
empty-partition levels plus 0.47 per unit of depth. Glymph measured 0.56 to 0.59 plus 0.47 to 0.54 from silent genes alone.

## Then every cell gets its own amount

Background is not one number per library. Each cell's own counts on its cluster's silent genes say whether it carries more or
less than the trend, and it gets its own multiplier, shrunk toward the trend when its evidence is thin. The same silent genes
constrain the model: it may not explain counts on them as expression.

One guard goes with that constraint. A cell with far more counts on those genes than background explains is held to it only if the
counts follow the measured background profile and are at most a fifth of the cell. Otherwise the cell expresses those genes
itself, as a rare cell type or a doublet would: it is left alone and flagged.

<p class="gl-take"><b>TAKE-HOME</b>Every library carries thousands of negative controls. Glymph reads the background off them.</p>

