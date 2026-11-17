#!/bin/bash
#SBATCH --job-name=glymph_metrics
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Score every tool on one library per task. Follow with 32_collect.sbatch for the tables and figure.
#
# Every folder under out/ that holds an output for the library is scored as a tool named after the
# folder (glymph, glymph_v0, cellbender, scar, soupx, ...). Outputs that do not exist are simply
# not scored, so this can be rerun as tools finish. Folders named *_failed are ignored;
# out/exp_* (development experiments) are scored by 31_metrics_exp.sbatch instead.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/metrics
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
tools=()
for d in out/*/; do
  t=$(basename "$d")
  case "$t" in metrics*|generic|deliverables|prep|envs|diag|glymph_4gpu*|exp_*|*_failed) continue ;; esac
  for f in "$d$name.h5ad" "$d$name.h5" "$d$name/${name}_filtered.h5" "$d$name.mtx"; do
    if [ -s "$f" ]; then tools+=(--tool "$t=$f"); break; fi   # -s is also true for a non-empty directory
  done
done
cells=()
case "$file" in *.h5ad) [ -s "out/prep/$name.cells.txt" ] && cells=(--cells "out/prep/$name.cells.txt") ;; esac
echo "$name: ${tools[*]} ${cells[*]}"
"$PY" code/benchmark/species_mix_metrics.py --name "$name" --raw "$file" "${cells[@]}" \
  "${tools[@]}" --out "out/metrics/$name.tsv" --per-cell out/metrics/per_cell
