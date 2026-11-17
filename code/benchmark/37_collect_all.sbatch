#!/bin/bash
#SBATCH --job-name=glymph_collect
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Tables of the all-tier benchmark into out/deliverables/ (collect_all_tiers.py). Rerun any time.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
"$PY" code/benchmark/collect_all_tiers.py "$PWD" config/datasets.tsv
cp out/glymph/*.report.json out/deliverables/ 2>/dev/null || true
ls -la out/deliverables | head -20
