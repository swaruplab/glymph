#!/bin/bash
#SBATCH --job-name=ambidose_hgmm
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Baseline: AmbiDose 0.5 (CPU) through its command line, given the same raw matrix and the same
# cell barcodes as the other baselines. Output: called cells with the denoised counts as X.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
ENV=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_ambidose
export PYTHONNOUSERSITE=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
mkdir -p out/ambidose
# AMBIDOSE_EXTRA (optional): extra flags, for example --full-typing. On libraries of about 9,000 cells
# and more its default fast graph hands scanpy 1.11 a sparse array type scanpy cannot read.
# Written as a 10x MTX folder: AmbiDose 0.5.15 fails while writing its own QC metadata to .h5ad
# on six of the seven libraries (TypeError in /uns/ambidose/trust), after the denoising is done.
rm -rf "out/ambidose/$name.mtx" "out/ambidose/$name.h5ad"
bash code/benchmark/profile.sh "out/ambidose/$name" "$ENV/bin/ambidose" denoise \
  --input "out/prep/$name.raw_v3.h5" --cell-barcodes "out/prep/$name.cells.txt" \
  --output "out/ambidose/$name.mtx" --output-format 10x-mtx --cells-only --report off \
  --summary-json "out/ambidose/$name.summary.json" --n-jobs 8 ${AMBIDOSE_EXTRA:-}
ls "out/ambidose/$name.mtx"
