#!/bin/bash
# Round M5 of 0.4.0 development (2026-11-18): 0.4.0.dev8. Kept from rounds M3 and M4: the swapped fraction read from gene
# contrast, now only where the contrast is clear and with a bound of 30% of depth (nuclei). Off: the level shared across
# clusters and the rule for background-like clusters (right on cell lines, wrong on tissue). Two runs at the defaults and
# two smaller false-positive allowances, on the 23 development libraries; then scoring.
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,32,33,34,42,44-55
mkdir -p out_old
for t in m1_pca_b m2_b glymph_032_rep2 m3_b m3_noswap m3_nobudget m3_fpr2 m4_b; do [ -d out/$t ] && mv out/$t out_old/; done
sub() { sbatch --parsable $X --array=$ROWS --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m5_a "")
b=$(sub m5_b "")
c=$(sub m5_fpr3 "--fpr=0.003")
d=$(sub m5_fpr2 "--fpr=0.002")
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b:$c:$d code/benchmark/36_score_array.sbatch)
echo "m5_a $a m5_b $b m5_fpr3 $c m5_fpr2 $d score $s"
