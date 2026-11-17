#!/bin/bash
# Download the tier 2 mouse kidney strain-mixture matrices (GSE218853, Janssen et al. 2023) straight
# onto HPC3 through the transfer node: raw and Cell Ranger filtered matrices of five libraries, 1.2 GB.
# Run from the MacBook (on VPN) or from the Studio.
set -euo pipefail
DEST=/dfs7/swaruplab/vswarup/compute/glymph/data/kidney_strainmix
TSV="$(cd "$(dirname "$0")/../.." && pwd)/config/benchmark_tier2_kidney.tsv"
ssh -o BatchMode=yes hpc3 "mkdir -p $DEST"
tail -n +2 "$TSV" | while IFS=$'\t' read -r name chem cr url furl rest; do
  echo "== $name"
  ssh -n -o BatchMode=yes hpc3-xfer "wget -nc -nv -P $DEST $url $furl"
done
ssh -o BatchMode=yes hpc3 "ls -la $DEST"
