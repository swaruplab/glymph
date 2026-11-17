#!/bin/bash
#SBATCH --job-name=glymph_var
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# One Glymph configuration on every library, written to out/<TAG>/ so it is scored as its own tool.
#   sbatch --export=ALL,TAG=glymph_b02,EXTRA="--bg-reward 0.02" 12_glymph_variant_array.sbatch
# TAG=glymph is the current default configuration.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
: "${TAG:?set TAG}"; EXTRA="${EXTRA:-}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
file="$DATA/$(basename "$(cut -f4 <<<"$line")")"
# Standard-format libraries carry the platform structure in obs: unit (barcodes sharing a soup,
# the round 1 well of split-pool data) and pool (barcodes amplified together, the sublibrary).
keys=$("$PY" - "$file" <<'EOF'
import sys, h5py
if sys.argv[1].endswith(".h5ad"):
    with h5py.File(sys.argv[1], "r") as f:
        obs = set(f["obs"].keys())
    print(" ".join(f"--{k}-key {k}" for k in ("unit", "pool") if k in obs))
EOF
)
mkdir -p "out/$TAG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
# A GPU that another process already fills makes the run fail with an out-of-memory error that is
# not Glymph's (hpc3-gpu-18-03 on 2026-11-17: 21.5 of 24 GB held by someone else's process).
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -gt 4000 ]; then echo "GPU already holds ${used} MiB from another process on $(hostname)" >&2; exit 75; fi
echo "TAG=$TAG EXTRA=$EXTRA keys=$keys"
# shellcheck disable=SC2086
bash code/benchmark/profile.sh "out/$TAG/$name" "$PY" -m glymph.cli run \
  --input "$file" --output "out/$TAG/$name.h5ad" --report "out/$TAG/$name.report.json" \
  --plot "out/$TAG/$name.rank.png" --device cuda:0 $keys $EXTRA
