#!/bin/bash
#SBATCH --job-name=kidney_metrics
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Tier 2: score every tool on one kidney library per task against the authors' genotype truth.
# Follow with 34_kidney_collect.sbatch. Folders are picked up as in 30_metrics.sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier2_kidney.tsv}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/metrics
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line"); kind=$(cut -f6 <<<"$line"); rep=$(cut -f7 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
tools=()
for d in out/*/; do
  t=$(basename "$d")
  case "$t" in metrics*|generic|deliverables|prep|envs|diag|glymph_4gpu*|exp_*|*_failed) continue ;; esac
  for f in "$d$name.h5ad" "$d$name.h5" "$d$name/${name}_filtered.h5" "$d$name.mtx"; do
    if [ -s "$f" ]; then tools+=(--tool "$t=$f"); break; fi
  done
done
echo "$name ($kind, $rep): ${tools[*]}"
"$PY" code/benchmark/kidney_metrics.py --name "$name" --replicate "$rep" --kind "$kind" --raw "$file" \
  --meta data/kidney_strainmix/cell_metadata.tsv.gz --markers data/kidney_strainmix/top10_PT_markers.txt \
  "${tools[@]}" --out "out/metrics/$name.tsv" --per-cell out/metrics/per_cell
