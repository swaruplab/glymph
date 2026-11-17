#!/bin/bash
#SBATCH --job-name=glymph_staridx
#SBATCH --account=vswarup_lab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# STAR index for the combined human + mouse genome (10x GRCh38-and-mm10-2020-A: contigs and gene
# ids carry a GRCh38_ or mm10_ prefix). Needed for the two species mixtures that are public only
# as reads: BD Rhapsody (GSE265962) and Parse Evercode (PRJEB67549).
# A sparse suffix array (--genomeSAsparseD 2) halves the index on disk and in memory; /dfs7 is 86% full.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
REF=/dfs7/swaruplab/shared_lab/lesolano/refrence/refdata-cellranger-atac-GRCh38-and-mm10-2020-A-2.0.0
IDX="${IDX:-/dfs7/swaruplab/vswarup/compute/glymph/ref/star_GRCh38_mm10_2020A_2710a_sparse2}"
module load star/2.7.10a
mkdir -p "$IDX"
zcat "$REF/genes/genes.gtf.gz" > "$IDX/genes.gtf"
bash code/benchmark/profile.sh logs/star_index STAR --runMode genomeGenerate --runThreadN "$SLURM_CPUS_PER_TASK" \
  --genomeDir "$IDX" --genomeFastaFiles "$REF/fasta/genome.fa" \
  --sjdbGTFfile "$IDX/genes.gtf" --sjdbOverhang 100 \
  --genomeSAsparseD 2 --limitGenomeGenerateRAM 100000000000 \
  --outFileNamePrefix "$IDX/build_"
grep -c . "$IDX/chrName.txt"; du -sh "$IDX"; cat "$IDX/genomeParameters.txt" | head -5
echo INDEX_DONE
