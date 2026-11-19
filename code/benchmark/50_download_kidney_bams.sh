#!/bin/bash
# Download the Cell Ranger alignment files of the mouse kidney strain mixture (GSE218853, Janssen et al. 2023) onto HPC3.
# The authors deposited possorted_genome_bam.bam for each library; the SRA keeps the original files in its public
# cloud buckets, which serve them far faster than ENA does. Every read carries CB, UB, GX and RE (exonic, intronic,
# intergenic), which is what the spliced / unspliced counts are made from (51_bam_splice_counts.py).
#
# Runs on a machine that can reach the HPC3 transfer node (the office Studio, in tmux):
#     bash 50_download_kidney_bams.sh
# One wget per file on the transfer node, all five in parallel, resumable (-c). 144 GB in all.
set -uo pipefail
DEST=/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix/bam
LOG="${LOG:-$HOME/LocalDocuments/compute/20261118-1245_glymph-040-dev/logs}"
mkdir -p "$LOG"
ssh -o BatchMode=yes hpc3 "mkdir -p $DEST"
get() {  # name, run, bucket
  url="https://sra-pub-src-$3.s3.amazonaws.com/$2/$1_possorted_genome_bam.bam.1"
  for try in 1 2 3 4 5; do
    ssh -o BatchMode=yes -o ServerAliveInterval=60 hpc3-xfer "wget -c -nv -O $DEST/$1_possorted_genome_bam.bam $url" >> "$LOG/$1.log" 2>&1 && { echo "$1 done $(date)" >> "$LOG/$1.log"; return 0; }
    echo "$1 attempt $try failed $(date)" >> "$LOG/$1.log"; sleep 30
  done
  return 1
}
get nuc2 SRR22426719 1 &
get nuc3 SRR22426718 2 &
get rep1 SRR22426722 2 &
get rep2 SRR22426721 2 &
get rep3 SRR22426720 2 &
wait
echo "all downloads returned $(date)" >> "$LOG/all.log"
