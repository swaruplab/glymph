#!/bin/bash
#SBATCH --job-name=glymph_expmet
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Score Glymph development experiments (out/exp_*) on one library per task, kept apart from the
# benchmark tables. Follow with:
#   python code/benchmark/summarize_experiments.py --run-dir . --metrics-dir <OUT>
# ONLY (optional): the output folders to score, for example ONLY="out/exp_v02/ out/glymph/".
# OUT: where the tables go (default out/metrics_exp).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
OUT="${OUT:-out/metrics_exp}"
mkdir -p "$OUT"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
tools=()
for d in ${ONLY:-out/exp_*/ out/glymph/ out/glymph_v0/}; do
  t=$(basename "$d")
  case "$t" in *_failed) continue ;; esac
  [ -s "$d$name.h5ad" ] && tools+=(--tool "$t=$d$name.h5ad")
done
"$PY" code/benchmark/species_mix_metrics.py --name "$name" --raw "$file" "${tools[@]}" --out "$OUT/$name.tsv" > /dev/null
echo "scored $name: ${tools[*]}"
