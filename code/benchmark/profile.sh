#!/bin/bash
# Run one command under the same profiler for every tool.
#
#   profile.sh <prefix> <command> [args...]
#
# Writes <prefix>.time.txt   /usr/bin/time -v: wall, user and system CPU seconds, percent of CPU, peak resident memory
#        <prefix>.gpu.csv    nvidia-smi sampled every 2 s for the job's GPUs: time, index, utilisation %, memory used MiB
#                            (only when a GPU is visible; the first line is the reading before the command starts,
#                            so memory already held by another process on a shared card can be subtracted)
#        <prefix>.host.txt   node name, CPU model, cores granted, GPU model
# Exits with the command's own exit code.
prefix=$1; shift
mkdir -p "$(dirname "$prefix")"
{
  echo "node $(hostname -s)"
  echo "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "cores_granted ${SLURM_CPUS_PER_TASK:-$(nproc)}"
  echo "mem_granted_mb ${SLURM_MEM_PER_NODE:-}"
} > "$prefix.host.txt"
smi=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)" >> "$prefix.host.txt"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits > "$prefix.gpu.csv" 2>/dev/null
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 >> "$prefix.gpu.csv" 2>/dev/null &
  smi=$!
fi
/usr/bin/time -v -o "$prefix.time.txt" "$@"
rc=$?
[ -n "$smi" ] && kill "$smi" 2>/dev/null
exit $rc
