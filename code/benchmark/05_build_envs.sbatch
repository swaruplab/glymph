#!/bin/bash
#SBATCH --job-name=glymph_envs
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Build the baseline environments inside a compute job (never on a login node).
#   task 1  glymph_bench_r         R with SoupX, decontX, Seurat (clustering), rhdf5
#   task 2  glymph_bench_scar      Python 3.10 with scAR (Novartis, from GitHub) and its own CUDA 12 PyTorch
#   task 3  glymph_bench_ambidose  Python 3.11 with AmbiDose from PyPI
# All three live next to the lab's other environments and can be deleted without touching anything else.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENVS=/dfs7/swaruplab/vswarup/bin/miniconda_envs
mkdir -p out/envs
set +u; source ~/conda_init.sh; set -u
export PIP_NO_CACHE_DIR=1
export PIP_CONFIG_FILE=/dev/null   # the user pip.conf adds an NVIDIA index that compute nodes cannot resolve
export PYTHONNOUSERSITE=1          # keep ~/.local packages out of the new environments

case "$SLURM_ARRAY_TASK_ID" in
1)
  env="$ENVS/glymph_bench_r"
  if [ ! -x "$env/bin/Rscript" ]; then
    if ! mamba create -y -p "$env" -c conda-forge -c bioconda \
        r-base r-soupx bioconductor-decontx r-seurat bioconductor-rhdf5 \
        bioconductor-singlecellexperiment r-matrix r-optparse; then
      echo "[envs] conda build of SoupX + decontX failed; falling back to installing them from R"
      case "$env" in */glymph_bench_r) rm -rf "$env" ;; esac
      mamba create -y -p "$env" -c conda-forge -c bioconda \
        r-base r-seurat r-biocmanager r-rstan r-remotes bioconductor-rhdf5 \
        bioconductor-singlecellexperiment bioconductor-scater r-matrix r-optparse compilers
      "$env/bin/Rscript" -e 'options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4); install.packages("SoupX"); BiocManager::install("decontX", update = FALSE, ask = FALSE)'
    fi
  fi
  "$env/bin/Rscript" -e 'cat("R", as.character(getRversion()), "\n"); for (p in c("SoupX", "decontX", "Seurat", "rhdf5", "Matrix")) cat(p, as.character(packageVersion(p)), "\n")' \
    | tee out/envs/glymph_bench_r.versions.txt
  ;;
2)
  # scAR is Novartis/scar on GitHub. The PyPI project called "scar" is an unrelated cloud tool
  # (the first build here installed it by mistake), so install from the repository archive.
  env="$ENVS/glymph_bench_scar"
  if [ -d "$env" ] && ! "$env/bin/python" -c "from scar import model" 2>/dev/null; then
    case "$env" in */glymph_bench_scar) rm -rf "$env" ;; esac
  fi
  if [ ! -x "$env/bin/python" ]; then
    mamba create -y -p "$env" -c conda-forge python=3.10 pip
  fi
  "$env/bin/pip" install torch
  "$env/bin/pip" install "https://github.com/Novartis/scar/archive/refs/heads/main.zip"
  "$env/bin/pip" install scanpy anndata h5py
  "$env/bin/python" code/benchmark/env_versions.py scar torch anndata scanpy numpy scipy pandas \
    --check "from scar import model, setup_anndata" | tee out/envs/glymph_bench_scar.versions.txt
  ;;
3)
  env="$ENVS/glymph_bench_ambidose"
  if [ ! -x "$env/bin/python" ]; then
    mamba create -y -p "$env" -c conda-forge python=3.11 pip
  fi
  "$env/bin/pip" install ambidose h5py
  "$env/bin/python" code/benchmark/env_versions.py ambidose anndata scanpy numpy scipy pandas torch \
    --check "import ambidose; print('ambidose_public', [n for n in dir(ambidose) if not n.startswith('_')])" \
    | tee out/envs/glymph_bench_ambidose.versions.txt
  ("$env/bin/ambidose" --help || "$env/bin/python" -m ambidose --help || true) 2>&1 | head -80 \
    > out/envs/glymph_bench_ambidose.help.txt
  ;;
esac
du -sh "$env" | tee -a "out/envs/$(basename "$env").versions.txt"
