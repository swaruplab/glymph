#!/bin/bash
# Confirmation run of Glymph 0.4.0rc1 (2026-11-18): every usable library of config/datasets.tsv, twice, scored beside
# CellBender 0.4.0, Glymph 0.3.2, scAR, SoupX and DecontX (outputs of the earlier benchmark runs, linked, not rerun).
#   bash submit_bench_040rc1.sh <run id>          (on the HPC3 login node; submits only)
set -euo pipefail
RUN=/dfs7/swaruplab/vswarup/compute/glymph/$1
cd "$RUN"
mkdir -p logs out
ln -sfn ../20261116-2209_glymph-alltier-bench/data data   # the kidney authors' metadata and marker list live there
ln -sfn ../../20261116-2209_glymph-alltier-bench/out/prep out/prep
ln -sfn ../../20261117-1342_glymph-cb040-bench/out/cellbender040 out/cellbender
ln -sfn ../../20261117-1342_glymph-cb040-bench/out/glymph out/glymph_032
for t in scar soupx decontx; do ln -sfn ../../20261116-2209_glymph-alltier-bench/out/$t out/$t; done
X=--exclude=hpc3-gpu-18-03,hpc3-gpu-18-04
ROWS=1-42,44-55
sub() { sbatch --parsable $X --array=$ROWS --job-name=$1 --export=ALL,TAG=$1,CONFIG=config/datasets.tsv,DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std,EXTRA= code/benchmark/12_glymph_variant_array.sbatch; }
a=$(sub glymph)
b=$(sub glymph_rep2)
s=$(sbatch --parsable --array=$ROWS --dependency=afterany:$a:$b code/benchmark/36_score_array.sbatch)
c=$(sbatch --parsable --dependency=afterany:$s code/benchmark/37_collect_all.sbatch)
t=$(sbatch --parsable -A vswarup_lab -p standard -c 8 --mem=32G -t 00:40:00 -J glymph_tests -o logs/%x_%j.out --wrap 'export PYTHONPATH=$PWD/src; P=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python; for t in test_superpose test_multires test_calibration test_splice test_background_match test_large_path test_smoke test_species_mix_e2e test_readers test_kidney_metrics; do echo == $t; $P tests/$t.py 2>&1 | grep -v -i warn | cut -c1-900 | tail -4; done')
echo "glymph $a glymph_rep2 $b score $s collect $c tests $t"
