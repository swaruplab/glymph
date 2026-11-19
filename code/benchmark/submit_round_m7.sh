#!/bin/bash
# Round M7 of 0.4.0 development (2026-11-18): 0.4.0.dev10 = 0.4.0rc1 with the tested pairs settled before the fit
# (calib_fixed_candidates). The 23 development libraries and three more that were looked at before and are not held
# out (bd_demo_pbmc, bd_spleen, bd_liver: rows 20, 21, 29), because the fault this round is about showed on the first.
# Two runs at the defaults and one with the old behaviour for the three added libraries; then scoring.
set -euo pipefail
cd /dfs7/swaruplab/vswarup/compute/glymph/20261118-1245_glymph-040-dev
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1,9,10,14,16,17,18,20,21,29,32,33,34,42,44-55
for t in m6_b m6_a m5_a m3_a m3_nofactor; do [ -d out/$t ] && mv out/$t out_old/; done
sub() { sbatch --parsable $X --array=$3 --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA="$2" code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub m7_a "" $ROWS)
b=$(sub m7_b "" $ROWS)
c=$(sub m6_fpr3 "--no-calib-fixed-candidates" 20,21,29)
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b:$c code/benchmark/36_score_array.sbatch)
t=$(sbatch --parsable -A vswarup_lab -p standard -c 8 --mem=32G -t 00:40:00 -J dev_t -o logs/dev_%j.out --wrap 'export PYTHONPATH=$PWD/src; P=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python; for t in test_superpose test_multires test_calibration test_splice test_background_match test_large_path test_smoke test_species_mix_e2e test_readers test_kidney_metrics; do echo == $t; $P tests/$t.py 2>&1 | grep -v -i warn | cut -c1-900 | tail -4; done')
echo "m7_a $a m7_b $b old_on_three $c score $s tests $t"
