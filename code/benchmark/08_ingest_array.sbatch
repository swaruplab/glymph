#!/bin/bash
#SBATCH --job-name=glymph_ingest
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One library per task: read the vendor format, write the standard .h5ad, cell list, summary and
# rank plot. Submit with --array=<rows of config/datasets.tsv, 1-based without the header>.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/datasets.tsv}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
name=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG" | cut -f1)
mkdir -p "$DATA/std" logs
bash code/benchmark/profile.sh "logs/ingest_$name" "$PY" code/benchmark/ingest_dataset.py \
  --config "$CONFIG" --name "$name" --data "$DATA" --out "$DATA/std"
