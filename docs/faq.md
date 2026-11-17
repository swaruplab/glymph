# Limits and FAQ

## Known limits

- **Very clean libraries.** With little background, only a handful of genes can be tested as silent, and the measured level can
  come out too high. On xenograft skin Glymph removes about 2 times the expected same-species background.
- **Run-to-run variation** on such libraries: up to 8 points of cross-species removal between two runs of the same code.
- **A single homogeneous population** has no silent genes. Glymph then keeps the empty-partition level and its learned swapped fraction.
- **It needs the empty partitions.** A matrix of called cells only cannot be run.
- **Doublets.** A doublet of two cell types is held to the silent genes of its main type, which strips the partner's exclusive genes.
  Such cells are flagged (`glymph_doublet_like`); drop them before calling doublets, or run with `--spare-doublets`.

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
