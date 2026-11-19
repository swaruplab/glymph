#!/bin/bash
# Round M6 of 0.4.0 development (2026-11-18): 0.4.0.dev9. As dev8, with the search for the swapped fraction made only
# where the depth-proportional background is at most 5% of depth (nuclei keep the soup profile) and the bound back at 5%.
# Two runs at the defaults and one with a false-positive allowance of 0.3%, on the 23 development libraries. The scoring
# that follows also writes the same-species truth with the split measured inside cells (columns ending in _measured).
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,32,33,34,42,44-55
mkdir -p out_old
for t in m5_b m5_fpr2 m5_fpr3 m4_a m1_pca_a; do [ -d out/$t ] && mv out/$t out_old/; done
sub() { sbatch --parsable $X --array=$ROWS --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m6_a "")
b=$(sub m6_b "")
c=$(sub m6_fpr3 "--fpr=0.003")
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b:$c code/benchmark/36_score_array.sbatch)
t=$(sbatch --parsable -A vswarup_lab -p standard -c 8 --mem=32G -t 00:40:00 -J dev_t -o logs/dev_%j.out --wrap 'export PYTHONPATH=$PWD/src; P=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python; for t in test_superpose test_multires test_calibration test_splice test_background_match test_large_path test_smoke test_species_mix_e2e test_readers test_kidney_metrics; do echo == $t; $P tests/$t.py 2>&1 | grep -v -i warn | cut -c1-900 | tail -4; done')
echo "m6_a $a m6_b $b m6_fpr3 $c score $s tests $t"
