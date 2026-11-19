#!/bin/bash
# Round M4 of 0.4.0 development (2026-11-18): 0.4.0.dev7 (gene contrast for the swapped fraction, level shared across
# clusters, background-like clusters take a larger cluster's silent genes) on the 23 development libraries, with
# variants of the false-positive allowance, the largest shared level and the tree. Scoring of rounds M3 and M4 follows.
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,32,33,34,42,44-55
sub() { sbatch --parsable $X --array=$ROWS --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m4_a "")
b=$(sub m4_b "")
c=$(sub m4_fpr2 "--fpr=0.002")
d=$(sub m4_fpr1 "--fpr=0.001")
e=$(sub m4_f2 "--silent-factor-max=2")
f=$(sub m4_notree "--no-silent-tree")
s=$(sbatch --parsable --partition=${SCORE_PARTITION:-standard} --array=$ROWS --dependency=afterany:$a:$b:$c:$d:$e:$f code/benchmark/36_score_array.sbatch)
echo "m4_a $a m4_b $b m4_fpr2 $c m4_fpr1 $d m4_f2 $e m4_notree $f score $s"
