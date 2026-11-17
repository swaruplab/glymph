#!/bin/bash
# Resource and scaling tests, beside the accuracy benchmark. Run on HPC3 from the run directory.
#   ROWS=1-42,44-55 bash code/benchmark/submit_resources.sh
# Every command already runs under profile.sh (wall and CPU time, peak RAM, GPU memory and
# utilisation sampled every 2 s, host and GPU model), so the accuracy jobs give per-library cost.
# This adds:
#   1. throughput    all libraries on one node with four A30s (glymph batch)
#   2. CPU only      Glymph without a GPU on a small, a medium and a large library (16 cores)
#   3. scaling       Glymph and CellBender on one library tiled 1, 2, 4 and 8 times (time and memory
#                    against library size; accuracy is not read from these)
#   4. GPU types     Glymph on the same medium library on every GPU model the lab account can use
set -euo pipefail
mkdir -p logs
: "${ROWS:?set ROWS, for example 1-42,44-55}"
D=/dfs7/swaruplab/vswarup/compute/glymph/data
B=code/benchmark
E="ALL,CONFIG=config/datasets.tsv,DATA=$D/std"
t=$(sbatch --parsable --export=$E,ROWS="$ROWS" $B/13_glymph_multigpu_all.sbatch)
cpu=""
for name in hgmm_1k_v3 kidney_rep3 hgmm_20k_v31_HT; do
  j=$(sbatch --parsable --job-name=glymph_cpu --export=ALL,INPUT=$D/std/$name.h5ad,NAME=$name $B/41_glymph_cpu.sbatch)
  cpu="$cpu $j"
done
sc=""
BASE=$D/hgmm/20k_hgmm_3p_HT_nextgem_Chromium_X_raw_feature_bc_matrix.h5
for tool in glymph cellbender; do
  for copies in 1 2 4 8; do
    j=$(sbatch --parsable --job-name=scale_${tool}_x$copies --export=ALL,TOOL=$tool,COPIES=$copies,BASE=$BASE,NAME=hgmm_20k $B/40_scaling.sbatch)
    sc="$sc $j"
  done
done
gp=""
for spec in gpu:V100 gpu:A100; do   # gpu32 (RTX6000, L40S) refuses the lab's GPU account: it wants a GPU32 account
  part=${spec%%:*}; type=${spec#*:}
  j=$(sbatch --parsable --partition=$part --gres=gpu:$type:1 --array=4 --job-name=glymph_$type \
        --export=ALL,TAG=glymph_gpu_$type,EXTRA=,CONFIG=config/benchmark_tier1.tsv,DATA=$D/hgmm $B/12_glymph_variant_array.sbatch)
  gp="$gp $j"
done
printf 'throughput_4gpu=%s cpu_only=%s scaling=%s gpu_types=%s\n' "$t" "$cpu" "$sc" "$gp"
