#!/bin/bash
#SBATCH --job-name=splice_counts
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Spliced / unspliced / ambiguous molecule counts of one kidney library from the authors' alignment file
# (50_download_kidney_bams.sh), on the barcodes and genes of its standard input file.
#   sbatch --job-name=splice_nuc2 --export=ALL,LIB=nuc2 51_bam_splice_counts.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${LIB:?set LIB, for example nuc2}"
B=/dfs7/swaruplab/vswarup/compute/glymph
BAM=$B/data/kidney_strainmix/bam/${LIB}_possorted_genome_bam.bam
GTF=/dfs7/swaruplab/vswarup/bin/CellRangerReference/refdata-gex-mm10-2020-A/genes/genes.gtf
PY=/dfs7/swaruplab/vswarup/bin/miniconda_envs/scanpy/bin/python
module load samtools/1.15.1
if [ -n "${MD5:-}" ]; then echo "$MD5  $BAM" | md5sum -c -; fi
[ -s "$BAM.bai" ] || samtools index -@ "$SLURM_CPUS_PER_TASK" "$BAM"
mkdir -p $B/data/splice
"$PY" code/benchmark/51_bam_splice_counts.py --bam "$BAM" --gtf "$GTF" --h5ad "$B/data/std/kidney_${LIB}.h5ad" \
  --out "$B/data/splice/kidney_${LIB}" --threads 16
ls -la $B/data/splice/kidney_${LIB}.*
