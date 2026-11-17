#!/bin/bash
# Download the tier 1 species-mixing matrices straight onto HPC3 through the transfer node.
# Run from the MacBook (on VPN) or from the Studio. About 1.2 GB in total.
set -euo pipefail
DEST=/dfs7/swaruplab/vswarup/compute/glymph/data/hgmm
TSV="$(cd "$(dirname "$0")/../.." && pwd)/config/benchmark_tier1.tsv"
ssh -o BatchMode=yes hpc3 "mkdir -p $DEST"
tail -n +2 "$TSV" | while IFS=$'\t' read -r name chem cr url rest; do
  echo "== $name"
  ssh -n -o BatchMode=yes hpc3-xfer "wget -nc -nv -P $DEST $url"
done
ssh -o BatchMode=yes hpc3 "ls -la $DEST"
