# Architecture

<p class="gl-kicker">A small autoencoder, calibrated by measurement</p>

## The pipeline

| Step | What happens |
|---|---|
| 1. Read | The whole matrix: cells and empty partitions |
| 2. Priors | Empty level, cell threshold and ambient profile from the depth distribution; a plateau mode for droplets, a knee mode where there is no plateau |
| 3. Train | The autoencoder with its count mixture |
| 4. Measure | Cluster the embedding, find silent genes, measure the three amounts and a multiplier per cell |
| 5. Refit | Train again briefly with the amounts fixed and silent genes excluded from expression |
| 6. Thin | Remove whole counts, gene by gene, from the entries where background is most probable |

Step 4 is the new part. Everything runs on one GPU, with the library held there as a sparse matrix.

## The model

A variational autoencoder built from fully connected layers. The encoder maps a barcode's counts to a low-dimensional embedding;
the decoder turns the embedding into the cell's own expression profile. The expected counts are

- the cell's own profile, from the decoder,
- plus the ambient profile of its unit, times the measured amount,
- plus the average cell profile of its pool, times the swapped fraction,

with a multinomial likelihood over the genes that have counts. The model is small on purpose: the difficulty is the amount of
background, not the capacity of the network.

## The output rule

For every gene, the number of counts to remove is the expected background of that gene summed over cells. They are taken from the
entries where background is most probable. A false-positive budget (0.4% by default) lets a gene give up a little more than
expected, and is the one dial that trades removal against counts kept. The result is an integer matrix in which no entry exceeds
the raw count.

## Three platforms

Platform structure enters as two labels per barcode.

| Platform | Unit (shares ambient RNA) | Pool (amplified together) | Priors from |
|---|---|---|---|
| 10x Chromium | the library | the library | the plateau of empty droplets |
| BD Rhapsody | the cartridge | the cartridge | the knee of the depth distribution |
| Parse Evercode | the first-round well | the sublibrary | the knee, per sublibrary |

Without vendor cell calls, Glymph's cell count on a BD Rhapsody spleen library is 8,061; the vendor pipeline calls 8,061.

<p class="gl-take"><b>TAKE-HOME</b>A small model is enough once the amount of background is no longer its job.</p>

