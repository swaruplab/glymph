#!/bin/bash
#SBATCH --job-name=glymph_scale
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Run-time and memory against library size: one tool on a library made COPIES times larger
# (make_scaled_library.py). For time and memory only, never accuracy.
#   sbatch --export=ALL,TOOL=glymph,COPIES=8,BASE=<raw.h5>,NAME=hgmm_20k 40_scaling.sbatch
# TOOL is glymph, cellbender (0.3.2 with the checkpoint fix) or cellbender040. Output: out/scaling/<TOOL>/<NAME>_x<COPIES>.{time.txt,gpu.csv,host.txt}
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${TOOL:?}" "${COPIES:?}" "${BASE:?}" "${NAME:?}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
tag="${NAME}_x${COPIES}"
mkdir -p out/scaling/inputs "out/scaling/$TOOL"
big="$PWD/out/scaling/inputs/$tag.h5"
[ -s "$big" ] || "$PY" code/benchmark/make_scaled_library.py --raw "$BASE" --copies "$COPIES" --out "$big"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$TOOL" in
glymph)
  bash code/benchmark/profile.sh "out/scaling/glymph/$tag" "$PY" -m glymph.cli run --input "$big" \
    --output "out/scaling/glymph/$tag.h5ad" --report "out/scaling/glymph/$tag.report.json" --device cuda:0 ${EXTRA:-}
  rm -f "out/scaling/glymph/$tag.h5ad"   # only the profile matters
  ;;
cellbender)
  FIX=/dfs7/swaruplab/vswarup/bin/cellbender_fix
  work="out/scaling/cellbender/$tag.work"; mkdir -p "$work"; cd "$work"
  bash "$SLURM_SUBMIT_DIR/code/benchmark/profile.sh" "../$tag" "$FIX/venv/bin/python" "$FIX/cellbender_safe.py" \
    remove-background --cuda --input "$big" --output "$tag.h5" --checkpoint-mins 20 > "$tag.cb.log" 2>&1 || true
  test -s "${tag}_filtered.h5" || { tail -30 "$tag.cb.log"; exit 1; }
  cd "$SLURM_SUBMIT_DIR"; rm -rf "$work"
  ;;
cellbender040)
  # CellBender 0.4.0 from its own environment (05c_build_cellbender040.sbatch); the plain command saves its checkpoint
  CB=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_cellbender040/bin/cellbender
  work="out/scaling/cellbender040/$tag.work"; mkdir -p "$work"; cd "$work"
  bash "$SLURM_SUBMIT_DIR/code/benchmark/profile.sh" "../$tag" "$CB" \
    remove-background --cuda --input "$big" --output "$tag.h5" --checkpoint-mins 20 > "$tag.cb.log" 2>&1 || true
  test -s "${tag}_filtered.h5" || { tail -30 "$tag.cb.log"; exit 1; }
  cd "$SLURM_SUBMIT_DIR"; rm -rf "$work"
  ;;
*) echo "unknown TOOL $TOOL"; exit 2 ;;
esac
