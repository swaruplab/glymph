#!/bin/bash
# Tier 2, mouse kidney strain mixture: submit every tool on the five libraries, then the scoring.
# Run on HPC3 from the run directory (it only calls sbatch, so the login node is fine).
# The matrices must already be in $DATA (01_download_tier2_kidney.sh).
set -euo pipefail
mkdir -p logs
export CONFIG=config/benchmark_tier2_kidney.tsv
export DATA=/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix
A="--array=1-5"
E="ALL,CONFIG=$CONFIG,DATA=$DATA"
B=code/benchmark
prep=$(sbatch --parsable $A --export=$E $B/03_prep_inputs.sbatch)
g=$(sbatch --parsable $A --job-name=glymph_kid --export=$E,TAG=glymph,EXTRA= $B/12_glymph_variant_array.sbatch)
g0=$(sbatch --parsable $A --job-name=glymph_kid --export=$E,TAG=glymph_fpr0,EXTRA="--fpr 0" $B/12_glymph_variant_array.sbatch)
g1=$(sbatch --parsable $A --job-name=glymph_kid --export=$E,TAG=glymph_fpr1,EXTRA="--fpr 0.01" $B/12_glymph_variant_array.sbatch)
m=$(sbatch --parsable --job-name=glymph_kid4 --export=$E $B/11_glymph_multigpu.sbatch)
c=$(sbatch --parsable $A --job-name=cellbender_kid --export=$E $B/20_cellbender_array.sbatch)
r=$(sbatch --parsable $A --job-name=rbase_kid --dependency=afterok:$prep --export=$E $B/21_r_baselines_array.sbatch)
s=$(sbatch --parsable $A --job-name=scar_kid --dependency=afterok:$prep --export=$E $B/22_scar_array.sbatch)
k=$(sbatch --parsable --dependency=afterany:$g:$g0:$g1:$c:$r:$s $B/33_kidney_metrics.sbatch)
z=$(sbatch --parsable --dependency=afterany:$k:$m $B/34_kidney_collect.sbatch)
printf 'prep=%s glymph=%s glymph_fpr0=%s glymph_fpr1=%s glymph_4gpu=%s cellbender=%s r_baselines=%s scar=%s metrics=%s collect=%s\n' \
  "$prep" "$g" "$g0" "$g1" "$m" "$c" "$r" "$s" "$k" "$z"
