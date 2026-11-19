#!/bin/bash
#SBATCH --job-name=splice_layers
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Standard input file of one kidney library with spliced / unspliced layers (data/std_splice/).
#   sbatch --job-name=layers_nuc2 --export=ALL,LIB=nuc2 52_add_splice_layers.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${LIB:?set LIB}"
B=/dfs7/swaruplab/vswarup/compute/glymph
/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python code/benchmark/52_add_splice_layers.py \
  --raw $B/data/std/kidney_${LIB}.h5ad --splice $B/data/splice/kidney_${LIB} --out $B/data/std_splice/kidney_${LIB}.h5ad
