#!/bin/bash
# Round M3 of 0.4.0 development (2026-11-18): gene contrast for the swapped fraction and the level shared across
# clusters, each alone and together, on the 23 development libraries. Run on the HPC3 login node (submits only).
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,32,33,34,42,44-55
sub() { sbatch --parsable $X --array=$ROWS --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m3_a "")
b=$(sub m3_b "")
c=$(sub m3_noswap "--no-swap-contrast")
d=$(sub m3_nofactor "--silent-factor-max=1")
e=$(sub m3_nobudget "--no-factor-budget")
f=$(sub m3_fpr2 "--fpr=0.002")
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b:$c:$d:$e:$f code/benchmark/36_score_array.sbatch)
echo "m3_a $a m3_b $b m3_noswap $c m3_nofactor $d m3_nobudget $e m3_fpr2 $f score $s"
