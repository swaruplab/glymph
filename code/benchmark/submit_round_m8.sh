#!/bin/bash
# Round M8 of 0.4.0 development (2026-11-18): 0.4.0.dev11. The background level is fitted under both rules for the tested
# pairs (following the fit, settled at the start); where they differ by more than half the lower one is kept
# (calib_cross_check). Round M7 showed that each rule alone runs away on one library. 26 libraries, two runs; scoring
# after the scoring of round M7 has finished.
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,20,21,29,32,33,34,42,44-55
sub() { sbatch --parsable $X --array=$3 --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m8_a "" $ROWS)
b=$(sub m8_b "" $ROWS)
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b:${PREV_SCORE:-1} code/benchmark/36_score_array.sbatch)
t=$(sbatch --parsable -A vswarup_lab -p standard -c 8 --mem=32G -t 00:40:00 -J dev_t -o logs/dev_%j.out --wrap 'export PYTHONPATH=$PWD/src; P=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python; for t in test_superpose test_multires test_calibration test_splice test_background_match test_large_path test_smoke test_species_mix_e2e test_readers test_kidney_metrics; do echo == $t; $P tests/$t.py 2>&1 | grep -v -i warn | cut -c1-900 | tail -4; done')
echo "m8_a $a m8_b $b score $s tests $t"
