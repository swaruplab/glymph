#!/bin/bash
# Download every file in config/datasets_manifest.tsv straight onto HPC3 through the transfer node.
# Meant to run on the office Studio in tmux (it never sleeps). Safe to rerun: wget -c resumes
# partial files and skips complete ones.
#
#   bash 02_download_manifest.sh <manifest.tsv> [parallel=3]
#
# Writes one line per file to stdout: OK or FAIL, the subfolder and the file name.
set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/sbin:/sbin:$PATH"
MAN=${1:?manifest}
PAR=${2:-3}
export BASE=/dfs7/swaruplab/vswarup/compute/glymph/data

dirs=$(tail -n +2 "$MAN" | cut -f3 | sort -u | sed "s#^#$BASE/#" | tr '\n' ' ')
ssh -o BatchMode=yes hpc3 "mkdir -p $dirs" || { echo "FAIL could not create folders on HPC3"; exit 1; }

fetch() {
  local platform dataset sub url out tgt name log rc
  IFS=$'\t' read -r platform dataset sub url out <<<"$1"
  name=${out:-${url##*/}}
  if [ -n "$out" ]; then tgt="-O $BASE/$sub/$out"; else tgt="-P $BASE/$sub"; fi
  for try in 1 2 3 4; do
    log=$(ssh -n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20 hpc3-xfer \
          "wget -c -nv --tries=5 --timeout=120 $tgt '$url'" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then echo "OK   $sub $name"; return 0; fi
    # wget exits 8 with "416 Requested Range Not Satisfiable" when the file is already complete
    if grep -q "416" <<<"$log"; then echo "OK   $sub $name (already complete)"; return 0; fi
    echo "retry $try $sub $name: $(tail -1 <<<"$log")"
    sleep 30
  done
  echo "FAIL $sub $name $url"
  return 1
}
export -f fetch

tail -n +2 "$MAN" | grep -v '^[[:space:]]*$' | tr '\n' '\0' | xargs -0 -P "$PAR" -I{} bash -c 'fetch "$1"' _ {}
echo "ALL DONE $(date '+%Y-%m-%d %H:%M %Z')"
