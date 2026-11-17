#!/bin/bash
#SBATCH --job-name=scar_hgmm
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Baseline: scAR (Novartis) with its tutorial settings, one library per A30.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_scar/bin/python
export PYTHONNOUSERSITE=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
mkdir -p out/scar
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash code/benchmark/profile.sh "out/scar/$name" "$PY" code/benchmark/run_scar.py \
  --input "out/prep/$name.raw_v3.h5" --cells "out/prep/$name.cells.txt" --output "out/scar/$name.h5ad"
