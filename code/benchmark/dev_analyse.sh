#!/bin/bash
# Development only: analyse outputs that run_debug_dump.sh left in <run dir>/out_dbg/<tag> (CPU; submit with sbatch).
#   sbatch -A vswarup_lab -p standard -c 4 --mem=48G -t 00:40:00 -o logs/dev_%j.out dev_analyse.sh <run dir> <tag> LIB [LIB ...]
# REF (default m2_a) is the scored run to compare with; its per-cell table gives the scored cells.
set -uo pipefail
RUN="$1"; TAG="$2"; shift 2
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
DATA="${DATA:-/dfs7/swaruplab/vswarup/compute/glymph/data/std}"
REF="${REF:-m2_a}"
cd "$RUN"
export PYTHONPATH="$RUN/dev2/src"
for L in "$@"; do
  echo "###### $L"
  for d in out_dbg/$TAG/$L.pool*.pt; do
    "$PY" dev2/code/benchmark/diag_silent_pairs.py --raw "$DATA/$L.h5ad" --dump "$d" --top 0 2>&1 | grep -E "clusters, level|amounts fitted|silent by the test" | cut -c1-330
  done
  cb=$(ls out/cellbender/$L.h5ad out/cellbender/$L.h5 out/cellbender/$L/${L}_filtered.h5 2>/dev/null | head -1)
  "$PY" dev2/code/benchmark/diag_cross_excess.py --raw "$DATA/$L.h5ad" --cells "out/prep/$L.cells.txt" --per-cell "out/metrics/per_cell/$L.$REF.tsv.gz" \
    --tool "new=out_dbg/$TAG/$L.h5ad" "$REF=out/$REF/$L.h5ad" "CellBender=$cb" --top 0 2>&1 | grep -E "both directions|^  [0-9s]|removed .* of all"
  "$PY" dev2/code/benchmark/diag_left_by_mask.py --raw "$DATA/$L.h5ad" --dump "out_dbg/$TAG/$L.pool0.pt" --out "out_dbg/$TAG/$L.h5ad" \
    --per-cell "out/metrics/per_cell/$L.$REF.tsv.gz" 2>&1 | grep -E "left |silent|overruled|not a candidate|rare, above|above the level|cells:" | grep -v "genes holding" | cut -c1-420
  "$PY" - "out_dbg/$TAG/$L.report.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: out.append(v)
            find(v, key, out)
    elif isinstance(o, list):
        for v in o: find(v, key, out)
    return out
for key in ("kappa", "kappa_depth", "ridge", "fit_rounds", "swap_share_of_depth_proportional", "genes_with_shared_level_above_1.25", "genes_with_shared_level_below_0.8",
            "pairs_silent_by_shared_level", "counts_silent_by_shared_level", "silent_counts_broad", "doublet_like_cells", "silent_excess_cells", "subcluster_gene_pairs_overruling_parent",
            "clusters_given_a_larger_clusters_silent_genes", "clusters_far_above_background_kept_apart", "subclusters_background_like_not_overruling"):
    v = find(r, key, [])
    if v: print(f"  report {key}: {v[0] if len(v) == 1 else v}"[:900])
PYEOF
done
echo "###### done"
