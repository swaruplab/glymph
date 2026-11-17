#!/usr/bin/env python3
"""Runs on the office Studio in tmux. Waits until every read file in config/read_files.tsv has its
full size on HPC3, restarting a transfer that has not grown for 10 minutes, then submits the four
STARsolo jobs and the ingestion of their output. The jobs verify the archive md5 themselves.

    python3 02b_wait_and_align.py <run dir on HPC3> <read_files.tsv>
"""
import csv
import os
import subprocess
import sys
import time

RD, TAB = sys.argv[1], sys.argv[2]
D = "/dfs7/swaruplab/vswarup/compute/glymph/data"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
files = list(csv.DictReader(open(TAB), delimiter="\t"))
last, stuck = {}, {}


def say(msg):
    print(time.strftime("%H:%M"), msg, flush=True)


while True:
    cmd = f"cd {D} && stat -c '%s %n' " + " ".join(f["path"] for f in files) + " 2>/dev/null"
    out = subprocess.run(SSH + ["hpc3", cmd], capture_output=True, text=True).stdout
    if not out.strip():
        say("could not reach HPC3")
        time.sleep(120)
        continue
    have = {line.split(" ", 1)[1]: int(line.split(" ", 1)[0]) for line in out.strip().splitlines()}
    todo = 0
    for f in files:
        p, size = f["path"], have.get(f["path"], 0)
        if size >= int(f["bytes"]):
            continue
        todo += 1
        stuck[p] = stuck.get(p, 0) + 1 if last.get(p) == size else 0
        last[p] = size
        if stuck[p] >= 5:
            say(f"restarting {p} at {size} of {f['bytes']}")
            subprocess.Popen(SSH + ["-o", "ServerAliveInterval=30", "hpc3-xfer",
                                    f"wget -c -nv --tries=5 --timeout=120 -P {D}/{os.path.dirname(p)} '{f['url']}'"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stuck[p] = 0
    say(f"{todo} read files still incomplete")
    if todo == 0:
        break
    time.sleep(120)

time.sleep(60)  # let the last writer close its file
F, Q, S = f"{D}/rhapsody/rock_roi/fastq", f"{D}/parse/PRJEB67549/fastq", "code/benchmark/07_starsolo_species_mix.sbatch"
jobs = [
    ("solo_bd1", "rhapsody", "bd_hgmm_mix1", f"{F}/SRR28817193", f"{D}/rhapsody/rock_roi/starsolo/mix1"),
    ("solo_bd2", "rhapsody", "bd_hgmm_mix2", f"{F}/SRR28830618", f"{D}/rhapsody/rock_roi/starsolo/mix2"),
    ("solo_p1", "parse", "parse_hgmm_sub1", f"{Q}/ERR12167393 {Q}/ERR12167395 {Q}/ERR12167397", f"{D}/parse/PRJEB67549/starsolo/sub1"),
    ("solo_p2", "parse", "parse_hgmm_sub2", f"{Q}/ERR12167394 {Q}/ERR12167396 {Q}/ERR12167398", f"{D}/parse/PRJEB67549/starsolo/sub2"),
]
ids = []
for jn, plat, name, pairs, outdir in jobs:
    cmd = (f"cd {RD} && sbatch --parsable --job-name={jn} "
           f"--export=ALL,PLATFORM={plat},NAME={name},PAIRS='{pairs}',OUT={outdir} {S}")
    r = subprocess.run(SSH + ["hpc3", cmd], capture_output=True, text=True)
    say(f"{jn}: {r.stdout.strip()} {r.stderr.strip()}")
    ids.append(r.stdout.strip().split(";")[0])
if all(i.isdigit() for i in ids):
    cmd = f"cd {RD} && sbatch --parsable --dependency=afterok:{':'.join(ids)} --array=18,19,32,33 code/benchmark/08_ingest_array.sbatch"
    r = subprocess.run(SSH + ["hpc3", cmd], capture_output=True, text=True)
    say(f"ingest: {r.stdout.strip()} {r.stderr.strip()}")
say("SUBMITTED " + " ".join(ids))
