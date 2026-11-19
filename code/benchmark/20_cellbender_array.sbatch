#!/bin/bash
#SBATCH --job-name=cellbender_hgmm
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Baseline: CellBender at defaults. By default the lab's 0.3.2 through its working launcher (below).
# Another version: set CB_CMD to the command that stands for "cellbender", and CB_TAG to its tool name:
#   CB_TAG=cellbender040 CB_CMD=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_cellbender040/bin/cellbender
# (0.4.0, built by 05c_build_cellbender040.sbatch).
#
# Launched through the lab's working recipe, /dfs7/swaruplab/vswarup/bin/cellbender_fix: every
# CellBender install on HPC3 trains and then dies at the checkpoint save with
# "cannot pickle 'weakref.ReferenceType' object". cellbender_safe.py fixes only that save;
# training and inference are unchanged. The first attempt here (job 59721468) hit exactly this.
# Success is the filtered matrix existing: the HTML report step fails (nbconvert is missing)
# and makes the exit code non-zero.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
FIX=/dfs7/swaruplab/vswarup/bin/cellbender_fix
PY=$FIX/venv/bin/python
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
# Cell Ranger v2 files hold one group per genome; CellBender gets them in the v3 layout
# written by 03_prep_inputs.sbatch. v3 files go in untouched.
case "$file" in
  *raw_gene_bc_matrices_h5.h5|*.h5ad) input="$PWD/out/prep/$name.raw_v3.h5" ;;   # never an .h5ad: CellBender pickles it into its checkpoint
  *) input="$file" ;;
esac
# CB_FROM_CELLS=1: give CellBender the number of cells and of droplets to analyse, as its
# documentation asks when the automatic priors fail (libraries without a droplet plateau).
# Expected cells = size of the benchmark's cell list; droplets = twice that, at least 5,000 more.
extra="${CB_EXTRA:-}"
# CB_FPRS="0.0 0.005 0.01": several false-positive budgets from one training run (CellBender writes
# one output per value). The 0.01 output, CellBender's default, stays the tool "cellbender"; the
# others are linked into out/cellbender_fpr<value>/ and scored as their own tools.
if [ -n "${CB_FPRS:-}" ]; then extra="$extra --fpr $CB_FPRS"; fi
if [ -n "${CB_FROM_CELLS:-}" ]; then
  n=$(grep -c . "out/prep/$name.cells.txt")
  tot=$(( n * 2 > n + 5000 ? n * 2 : n + 5000 ))
  extra="$extra --expected-cells $n --total-droplets-included $tot"
fi
work="out/${CB_TAG:-cellbender}/$name"
mkdir -p "$work"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$work"   # CellBender writes its checkpoint into the working directory
cmd="${CB_CMD:-$PY $FIX/cellbender_safe.py}"
echo "command: $cmd"
# shellcheck disable=SC2086
bash "$SLURM_SUBMIT_DIR/code/benchmark/profile.sh" "../$name" $cmd remove-background --cuda \
  --input "$input" --output "$name.h5" --checkpoint-mins 20 $extra > "$name.cb.log" 2>&1 || true
if [ -n "${CB_FPRS:-}" ]; then
  for f in "${name}"_FPR_*_filtered.h5; do
    [ -e "$f" ] || continue
    v=${f#"${name}"_FPR_}; v=${v%_filtered.h5}
    if awk -v v="$v" 'BEGIN { exit !(v + 0 == 0.01) }'; then
      ln -sf "$f" "${name}_filtered.h5"
    else
      tag=$(awk -v v="$v" 'BEGIN { printf "%g", v * 100 }' | tr -d .)
      d="$SLURM_SUBMIT_DIR/out/${CB_TAG:-cellbender}_fpr$tag/$name"
      mkdir -p "$d"
      ln -sf "$PWD/$f" "$d/${name}_filtered.h5"
    fi
  done
fi
test -s "${name}_filtered.h5" || { tail -40 "$name.cb.log"; exit 1; }
rm -f ckpt.tar.gz   # about 1 GB each, not needed once the output exists
tail -3 "$name.cb.log"
echo "done: $work/${name}_filtered.h5"
