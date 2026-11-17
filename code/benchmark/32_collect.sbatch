#!/bin/bash
#SBATCH --job-name=glymph_collect
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Gather the per-library metric tables, run times and Glymph reports into out/deliverables.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
mkdir -p out/deliverables
"$PY" code/benchmark/collect_results.py --run-dir . --out out/deliverables
cp out/glymph/*.rank.png out/deliverables/ 2>/dev/null || true
for f in out/glymph/*.report.json; do
  [ -s "$f" ] && cp "$f" "out/deliverables/glymph.$(basename "$f")"
done
