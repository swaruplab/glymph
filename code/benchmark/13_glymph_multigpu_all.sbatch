#!/bin/bash
#SBATCH --job-name=glymph_4gpu_all
#SBATCH --account=vswarup_lab_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Throughput: every library of the table (ROWS, as in submit_all_tiers.sh) on one node with four
# A30s, one library per GPU at a time (glymph batch). The number to read is the wall time in
# out/glymph_4gpu/batch.time.txt and wall_s in out/glymph_4gpu/batch_summary.json; it compares with
# the sum of the single-GPU times of any tool. Outputs duplicate out/glymph and are not scored.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/datasets.tsv}"
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/std}"
: "${ROWS:?set ROWS, for example 1-42,44-55}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$PWD/src"
mkdir -p out/glymph_4gpu
"$PY" - "$CONFIG" "$DATA" "$ROWS" > out/glymph_4gpu/manifest.tsv <<'PYEOF'
import csv, sys, h5py
cfg, data, rows = sys.argv[1:4]
want = set()
for part in rows.split(","):
    a, _, b = part.partition("-")
    want.update(range(int(a), int(b or a) + 1))
print("name\tinput\tunit_key\tpool_key")
for i, r in enumerate(csv.DictReader(open(cfg), delimiter="\t"), start=1):
    if i not in want:
        continue
    path = f"{data}/{r['file']}"
    with h5py.File(path, "r") as f:
        obs = set(f["obs"].keys())
    print("\t".join([r["name"], path, "unit" if "unit" in obs else "", "pool" if "pool" in obs else ""]))
PYEOF
wc -l out/glymph_4gpu/manifest.tsv
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash code/benchmark/profile.sh out/glymph_4gpu/batch "$PY" -m glymph.cli batch \
  --manifest out/glymph_4gpu/manifest.tsv --outdir out/glymph_4gpu --devices cuda:0,cuda:1,cuda:2,cuda:3 ${EXTRA:-}
