#!/bin/bash
# Development only: run Glymph on libraries with the silent-test dump on, then analyse it.
#   bash run_debug_dump.sh <source tree with src/ and code/> <out dir> LIB [LIB ...]     (on a GPU node)
set -euo pipefail
SRC="$1"; OUT="$2"; shift 2
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/std}"
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
export PYTHONPATH="$SRC/src"
mkdir -p "$OUT"
for L in "$@"; do
  file="$DATA/$L.h5ad"
  keys=$("$PY" - "$file" <<'PYEOF'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    obs = set(f["obs"].keys())
print(" ".join(f"--{k}-key {k}" for k in ("unit", "pool") if k in obs))
PYEOF
)
  # shellcheck disable=SC2086
  GLYMPH_DEBUG_DUMP="$OUT/$L" "$PY" -m glymph.cli run --input "$file" --output "$OUT/$L.h5ad" --report "$OUT/$L.report.json" \
    --device cuda:0 $keys ${EXTRA:-} 2>&1 | grep -v -i "warn\|epoch" | tail -4
  if [ "${ANALYSE:-1}" = 1 ]; then
    for d in "$OUT/$L".pool*.pt; do
      "$PY" "$SRC/code/benchmark/diag_silent_pairs.py" --raw "$file" --dump "$d" 2>&1 | grep -v -i warn || true
    done
  fi
  echo "======"
done
