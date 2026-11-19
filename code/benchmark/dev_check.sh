#!/bin/bash
# Development only: run the working copy of Glymph (<run dir>/dev2) on a few libraries with the silent-test dump on,
# then analyse the outputs (dev_analyse.sh).   sbatch ... dev_check.sh <run dir> <tag> LIB [LIB ...]   (GPU node)
set -uo pipefail
RUN="$1"; TAG="$2"; shift 2
cd "$RUN"
ANALYSE=0 bash dev2/code/benchmark/run_debug_dump.sh "$RUN/dev2" "$RUN/out_dbg/$TAG" "$@" 2>&1 | grep -E "budget from|removed [0-9.]+% of|Traceback|Error"
bash dev2/code/benchmark/dev_analyse.sh "$RUN" "$TAG" "$@"
