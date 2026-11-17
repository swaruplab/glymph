#!/bin/bash
#SBATCH --job-name=glymph_cpu
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Glymph with no GPU at all, 16 CPU cores: what a lab without a GPU would wait.
#   sbatch --export=ALL,INPUT=<raw.h5>,NAME=hgmm_10k_v3 41_glymph_cpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${INPUT:?}" "${NAME:?}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p out/scaling/glymph_cpu
bash code/benchmark/profile.sh "out/scaling/glymph_cpu/$NAME" "$PY" -m glymph.cli run --input "$INPUT" \
  --output "out/scaling/glymph_cpu/$NAME.h5ad" --report "out/scaling/glymph_cpu/$NAME.report.json" --device cpu
