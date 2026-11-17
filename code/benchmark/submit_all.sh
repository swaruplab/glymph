#!/bin/bash
# Run on HPC3 from the run directory (it only calls sbatch, so the login node is fine).
set -euo pipefail
mkdir -p logs
g=$(sbatch --parsable code/benchmark/10_glymph_array.sbatch)
m=$(sbatch --parsable code/benchmark/11_glymph_multigpu.sbatch)
c=$(sbatch --parsable code/benchmark/20_cellbender_array.sbatch)
s=$(sbatch --parsable --dependency=afterany:$g:$c code/benchmark/30_metrics.sbatch)
echo "glymph_array=$g glymph_4gpu=$m cellbender_array=$c metrics=$s"
