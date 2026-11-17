#!/bin/bash
# Wait until none of the given Slurm job ids is pending or running, or until the time limit.
#   wait_for_jobs.sh <max minutes> <job id> [<job id> ...]
# Polls the scheduler every 5 minutes. Meant to run in the background on the MacBook.
max=$1; shift
ids=$(IFS=,; echo "$*")
end=$(( $(date +%s) + max * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  left=$(ssh -o BatchMode=yes -o ConnectTimeout=20 hpc3 "squeue -h -j $ids -o '%i %j %T' 2>/dev/null" 2>/dev/null)
  rc=$?
  n=$(printf '%s\n' "$left" | grep -c .)
  echo "$(date '+%H:%M') still queued or running: $n"
  if [ $rc -eq 0 ] && [ "$n" -eq 0 ]; then echo "all done"; exit 0; fi
  sleep 300
done
echo "time limit reached; still active:"; printf '%s\n' "$left"
