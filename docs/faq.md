# Limits and FAQ

## Known limits

- **Very clean libraries.** With little background, only a handful of genes can be tested as silent, and the measured level can
  come out too high. On xenograft skin Glymph removes about 1.2 times the expected same-species background
  (a second run: 1.7).
- **Run-to-run variation** on such libraries. Two runs of the same code differ by a median of 0.3 points of cross-species removal
  across the benchmark, but by 5 to 21 points on four very clean or shallow libraries. A run can also find no estimate at all there;
  it then falls back to the learned values and removes less. The report says which happened (`swap_calibration`).
- **A single homogeneous population** has no silent genes. Glymph then keeps the empty-partition level and its learned swapped fraction.
- **It needs the empty partitions.** A matrix of called cells only cannot be run.
- **Rare cell types and doublets.** A cell that sits in a cluster of another kind has many counts on that cluster's silent genes.
  Glymph leaves such a cell alone when those counts do not follow the background profile or are more than a fifth of the cell, and
  flags it (`glymph_doublet_like`); it is still cleaned at the ordinary level. A flagged cell is a rare cell type or a doublet:
  look at it before discarding it. A subtler subtype, with less than a fifth of its counts on such genes, can still lose them.

## Does it need cell type labels or marker lists?

No. Clusters come from the model's own embedding.

## Does it change counts upward?

Never. The output is the input minus whole counts.

## Which budget should I use?

The default (0.4%). A larger `--fpr` removes more and keeps fewer of the cells' own counts; the benchmark page shows the trade-off.

## How was it tested?

On 54 libraries from three platforms against CellBender, scAR, SoupX and DecontX, with species, genotype and tissue truth.
Twelve libraries were never used in development. See the [benchmark](benchmark.md).

## What is the licence?

MIT. See `LICENSE` in the repository.

## Why "Glymph"?

After the glymphatic system, which clears waste from the brain.
