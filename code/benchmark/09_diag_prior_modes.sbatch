#!/bin/bash
#SBATCH --job-name=glymph_priordiag
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Plateau candidate, homogeneity and knee for every library, in four chunks.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
D=/dfs7/swaruplab/vswarup/compute/glymph/data
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
specs=()
# the standard-format copy of every library in the table (tier 1 and kidney included)
while IFS=$'\t' read -r name _ _ file _; do [ "$name" = name ] || specs+=("$name=$D/std/$file"); done < config/datasets.tsv
mine=()
for i in "${!specs[@]}"; do [ $((i % 4 + 1)) -eq "$SLURM_ARRAY_TASK_ID" ] && mine+=("${specs[$i]}"); done
mkdir -p out/diag
"$PY" code/benchmark/${DIAG:-diag_prior_modes.py} ${DIAG:+} $([ -z "${DIAG:-}" ] && echo "out/diag/prior_modes_${SLURM_ARRAY_TASK_ID}.tsv") "${mine[@]}"
