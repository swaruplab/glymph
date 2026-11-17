#!/bin/bash
#SBATCH --job-name=glymph_4gpu
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# All seven libraries on one 4-GPU node, one library per GPU at a time.
# The number to read is wall_s in out/glymph_4gpu/batch_summary.json.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/glymph_4gpu
printf 'name\tinput\n' > out/glymph_4gpu/manifest.tsv
tail -n +2 "$CONFIG" | while IFS=$'\t' read -r name chem cr url rest; do
  printf '%s\t%s\n' "$name" "$DATA/$(basename "$url")" >> out/glymph_4gpu/manifest.tsv
done
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash code/benchmark/profile.sh out/glymph_4gpu/batch "$PY" -m glymph.cli batch \
  --manifest out/glymph_4gpu/manifest.tsv --outdir out/glymph_4gpu --devices cuda:0,cuda:1,cuda:2,cuda:3
