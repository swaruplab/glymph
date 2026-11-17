#!/bin/bash
#SBATCH --job-name=rbase_hgmm
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Baselines on CPU: SoupX, DecontX, and DecontX with empty droplets as background.
# Each tool is timed on its own, including reading the input, as Glymph and CellBender are.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${CONFIG:-config/benchmark_tier1.tsv}"   # library table; set CONFIG and DATA for another tier
ENV=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_r
export PATH="$ENV/bin:$PATH"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG")
name=$(cut -f1 <<<"$line")
status=0
for tool in ${TOOLS:-soupx decontx decontx_bg}; do
  mkdir -p "out/$tool"
  # SKIP_DONE=1: a second pass that only fills in outputs the first pass did not write
  # (package loading from the shared filesystem fails when dozens of R processes start together).
  if [ -n "${SKIP_DONE:-}" ] && [ -s "out/$tool/$name.h5" ]; then continue; fi
  ok=0
  for attempt in 1 2 3 4 5; do   # package loading from the shared filesystem fails now and then; retry
    if bash code/benchmark/profile.sh "out/$tool/$name" "$ENV/bin/Rscript" code/benchmark/run_r_baselines.R \
        --tool "$tool" --input "out/prep/$name.raw_v3.h5" --cells "out/prep/$name.cells.txt" \
        --output "out/$tool/$name.h5"; then ok=1; break; fi
    echo "[baseline] $tool attempt $attempt failed on $name"; sleep 20
  done
  [ "$ok" = 1 ] || { echo "[baseline] $tool FAILED on $name"; status=1; }
done
exit $status
