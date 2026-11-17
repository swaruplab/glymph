#!/bin/bash
#SBATCH --job-name=glymph_score
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Score every tool on one library of config/datasets.tsv per task, by the kind of truth it has:
#   species    cross-species counts (species_mix_metrics.py)        -> out/metrics/<name>.tsv
#              (alignment-artifact genes masked; out/metrics_unmasked/<name>.tsv has every gene)
#   genotype   the kidney authors' per-cell truth (kidney_metrics.py) -> out/metrics/<name>.tsv
#   all        integrity, amount, exclusive-marker leakage (generic_metrics.py) -> out/generic/<name>.tsv
# Every folder under out/ that holds an output for the library counts as a tool. Rerun any time.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/datasets.tsv}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/std}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/metrics out/metrics_unmasked out/generic
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line"); kind=$(cut -f5 <<<"$line"); truth=$(cut -f6 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
tools=()
for d in out/*/; do
  t=$(basename "$d")
  case "$t" in metrics*|generic|deliverables|prep|envs|diag|scaling|glymph_4gpu*|glymph_cpu*|exp_*|*_failed) continue ;; esac
  for f in "$d$name.h5ad" "$d$name.h5" "$d$name/${name}_filtered.h5" "$d$name.mtx"; do
    if [ -s "$f" ]; then tools+=(--tool "$t=$f"); break; fi
  done
done
echo "$name ($kind, truth: $truth): ${tools[*]}"
case "$truth" in
  species)
    "$PY" code/benchmark/species_mix_metrics.py --name "$name" --raw "$file" --cells "out/prep/$name.cells.txt" \
      "${tools[@]}" --out "out/metrics/$name.tsv" --per-cell out/metrics/per_cell --artifacts-out "out/metrics/artifacts/$name.tsv"
    # the same scores with every gene counted, alignment artifacts included, for the record
    "$PY" code/benchmark/species_mix_metrics.py --name "$name" --raw "$file" --cells "out/prep/$name.cells.txt" \
      "${tools[@]}" --out "out/metrics_unmasked/$name.tsv" --no-mask-artifacts ;;
  genotype)
    "$PY" code/benchmark/kidney_metrics.py --name "$name" --replicate "${name#kidney_}" --kind "$kind" --raw "$file" \
      --meta data/kidney_strainmix/cell_metadata.tsv.gz --markers data/kidney_strainmix/top10_PT_markers.txt \
      "${tools[@]}" --out "out/metrics/$name.tsv" --per-cell out/metrics/per_cell ;;
esac
"$PY" code/benchmark/generic_metrics.py --name "$name" --raw "$file" --cells "out/prep/$name.cells.txt" \
  "${tools[@]}" --out "out/generic/$name.tsv"
