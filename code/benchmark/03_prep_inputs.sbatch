#!/bin/bash
#SBATCH --job-name=glymph_prep
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Shared inputs for the baseline tools: raw matrix in one HDF5 layout, plus cell barcodes.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
# A table with a filtered_url column (tier 2) hands the baselines Cell Ranger's own cell calls.
fcol=$(head -1 "$CONFIG" | tr '\t' '\n' | grep -n -x filtered_url | cut -d: -f1 || true)
extra=()
if [ -n "$fcol" ]; then extra=(--filtered "$DATA/$(basename "$(cut -f"$fcol" <<<"$line")")"); fi
# Standard-format libraries (config/datasets.tsv): the vendor's cell calls when ingestion found
# them, otherwise barcodes above the gap between cells and background.
case "$file" in *.h5ad) if [ -s "$DATA/$name.cells.txt" ]; then extra=(--cells "$DATA/$name.cells.txt"); else extra=(--knee); fi ;; esac
"$PY" code/benchmark/prep_inputs.py --name "$name" --raw "$file" --outdir out/prep "${extra[@]}"
