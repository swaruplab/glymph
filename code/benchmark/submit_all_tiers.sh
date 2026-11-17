#!/bin/bash
# Every tier in one go: all usable libraries of config/datasets.tsv, every tool, profiled, then scored.
# Run on HPC3 from the run directory (it only calls sbatch). The standard-format inputs must exist
# in $DATA (08_ingest_array.sbatch). ROWS lists the table rows to run, 1-based without the header.
#   ROWS=1-42,44-55 bash code/benchmark/submit_all_tiers.sh
set -euo pipefail
mkdir -p logs
export CONFIG=config/datasets.tsv
export DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/std
: "${ROWS:?set ROWS, for example 1-42,44-55}"
A="--array=$ROWS"
E="ALL,CONFIG=$CONFIG,DATA=$DATA"
B=code/benchmark
prep=$(sbatch --parsable $A --export=$E $B/03_prep_inputs.sbatch)
# Glymph: default settings (false-positive budget 0.4%), and budgets of 0, 0.2% and 1% for the
# trade-off curve.
g=$(sbatch --parsable $A --time=02:00:00 --mem=64G --job-name=glymph_all --export=$E,TAG=glymph,EXTRA= $B/12_glymph_variant_array.sbatch)
g0=$(sbatch --parsable $A --time=02:00:00 --mem=64G --job-name=glymph_all --export=$E,TAG=glymph_fpr0,EXTRA="--fpr 0" $B/12_glymph_variant_array.sbatch)
g2=$(sbatch --parsable $A --time=02:00:00 --mem=64G --job-name=glymph_all --export=$E,TAG=glymph_fpr02,EXTRA="--fpr 0.002" $B/12_glymph_variant_array.sbatch)
g1=$(sbatch --parsable $A --time=02:00:00 --mem=64G --job-name=glymph_all --export=$E,TAG=glymph_fpr1,EXTRA="--fpr 0.01" $B/12_glymph_variant_array.sbatch)
# Baselines. CellBender and scAR on the same GPU type as Glymph; SoupX and DecontX on CPU.
# One CellBender training run per library writes three false-positive budgets: 0.01 (its default,
# the tool "cellbender"), 0.005 and 0, so its trade-off curve can be set beside Glymph's.
c=$(sbatch --parsable $A --job-name=cellbender_all --dependency=afterok:$prep --export="$E,CB_FPRS=0.0 0.005 0.01" $B/20_cellbender_array.sbatch)
r=$(sbatch --parsable $A --mem=120G --job-name=rbase_all --dependency=afterok:$prep --export=$E $B/21_r_baselines_array.sbatch)
s=$(sbatch --parsable $A --job-name=scar_all --dependency=afterok:$prep --export=$E $B/22_scar_array.sbatch)
k=$(sbatch --parsable $A --dependency=afterany:$g:$g0:$g2:$g1:$c:$r:$s --export=$E $B/36_score_array.sbatch)
z=$(sbatch --parsable --dependency=afterany:$k $B/37_collect_all.sbatch)
printf 'collect=%s ' "$z"
printf 'prep=%s glymph=%s glymph_fpr0=%s glymph_fpr02=%s glymph_fpr1=%s cellbender=%s r_baselines=%s scar=%s score=%s\n' \
  "$prep" "$g" "$g0" "$g2" "$g1" "$c" "$r" "$s" "$k"
