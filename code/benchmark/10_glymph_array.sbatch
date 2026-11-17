#!/bin/bash
#SBATCH --job-name=glymph_hgmm
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Glymph, one library per GPU. Same GPU type as the CellBender array so times compare.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
mkdir -p out/glymph
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash code/benchmark/profile.sh "out/glymph/$name" "$PY" -m glymph.cli run \
  --input "$file" --output "out/glymph/$name.h5ad" --report "out/glymph/$name.report.json" \
  --plot "out/glymph/$name.rank.png" --device cuda:0
