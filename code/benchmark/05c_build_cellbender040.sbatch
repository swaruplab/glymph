#!/bin/bash
#SBATCH --job-name=build_cb040
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CellBender 0.4.0 in its own environment (the lab's CellBender is 0.3.2). Built inside a compute job,
# never on a login node. Python 3.11 from conda-forge, CellBender and its pinned PyTorch from PyPI
# (the wheels bring their own CUDA runtime).
#   sbatch code/benchmark/05c_build_cellbender040.sbatch
set -euo pipefail
ENV=/dfs7/swaruplab/vswarup/bin/miniconda_envs/glymph_bench_cellbender040
source ~/conda_init.sh
if [ ! -x "$ENV/bin/python" ]; then
  mamba create -y -p "$ENV" -c conda-forge python=3.11 pip
fi
"$ENV/bin/pip" install --no-cache-dir "cellbender==0.4.0"
"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch
for p in ("cellbender", "torch", "pyro-ppl", "numpy", "scipy", "tables", "anndata"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "not installed")
print("cuda build:", torch.version.cuda)
PY
"$ENV/bin/cellbender" remove-background --help | head -5
echo "built $ENV"
