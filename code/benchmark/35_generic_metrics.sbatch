#!/bin/bash
#SBATCH --job-name=glymph_generic
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Truth-free scores (integrity, amount removed, exclusive-marker leakage) for one library per task,
# every tool folder under out/. Submit with --array=<rows of the library table>.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/datasets.tsv}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/std}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/generic
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
tools=()
for d in out/*/; do
  t=$(basename "$d")
  case "$t" in metrics*|generic|deliverables|prep|envs|diag|glymph_4gpu*|exp_*|*_failed) continue ;; esac
  for f in "$d$name.h5ad" "$d$name.h5" "$d$name/${name}_filtered.h5" "$d$name.mtx"; do
    if [ -s "$f" ]; then tools+=(--tool "$t=$f"); break; fi
  done
done
echo "$name: ${tools[*]}"
"$PY" code/benchmark/generic_metrics.py --name "$name" --raw "$file" --cells "out/prep/$name.cells.txt" \
  "${tools[@]}" --out "out/generic/$name.tsv"
