#!/usr/bin/env python3
"""Build dashboard/tracker.json, the one file behind the project dashboard.

Run on the MacBook (standard library only):

    python3 code/dashboard/build_tracker.py

It reads
    runs/index.tsv                      one row per run (also rewritten as runs/INDEX.md)
    runs/<run_id>/jobs.tsv              Slurm job ids of each run
    runs/<run_id>/decisions.tsv         dated log of decisions and events
    config/tools.tsv                    tools in the benchmark and where they are installed
    config/published_reference.tsv      numbers quoted from the published benchmarks
and asks HPC3 for
    sacct                               state, elapsed time and peak memory of every job
    out/deliverables, out/envs          metric tables and environment versions (rsync, small files)

Numbers are rounded here to the precision the dashboard prints, so the page shows exactly
what this file holds.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HPC3_BASE = "/dfs7/swaruplab/vswarup/compute/glymph"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
# Off campus without the VPN: GLYMPH_HPC3_VIA=studio-work asks the scheduler through that host (it is on campus and has
# its own HPC3 key). Result files are then not fetched here; pull them with code/hpc_relay/h3pull.
VIA = os.environ.get("GLYMPH_HPC3_VIA", "")
HIDE_TOOLS = {"glymph_map"}  # an output-rule experiment that was scored once with the main tools


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if any((v or "").strip() for v in r.values())]


def num(x, nd=None, scale=1.0):
    try:
        v = float(x) * scale
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if nd is None:
        return v
    v = round(v, nd)
    return int(v) if nd == 0 else v


def rss_gb(s):
    s = (s or "").strip()
    if not s:
        return None
    mult = {"K": 1 / 1e6, "M": 1 / 1e3, "G": 1.0, "T": 1e3}.get(s[-1])
    try:
        return round(float(s[:-1]) * mult, 1) if mult else round(float(s) / 1e9, 1)
    except ValueError:
        return None


def hms(seconds):
    if seconds is None:
        return ""
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def sacct(job_ids):
    if not job_ids:
        return []
    fmt = "JobID,JobName,State,ElapsedRaw,Start,End,MaxRSS,NodeList,ExitCode"
    remote = f"sacct -j {','.join(job_ids)} -P -n --format={fmt}"
    cmd = SSH + ([VIA, f"ssh -o BatchMode=yes -o ConnectTimeout=20 hpc3 '{remote}'"] if VIA else ["hpc3", remote])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print("[tracker] sacct failed:", out.stderr.strip()[:300], file=sys.stderr)
        return None
    return [dict(zip(fmt.split(","), line.split("|"))) for line in out.stdout.splitlines() if line.strip()]


def expand_tasks(spec):
    """'59721466_[4-7]' -> ['4','5','6','7']; '59721466_3' -> ['3']; '59721467' -> ['']."""
    if "_" not in spec:
        return [""]
    t = spec.split("_", 1)[1]
    if not t.startswith("["):
        return [t]
    out = []
    for part in t.strip("[]").split("%")[0].split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return out


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    libs = [r["name"] for r in read_tsv(os.path.join(ROOT, "config", "benchmark_tier1.tsv"))]
    runs = read_tsv(os.path.join(ROOT, "runs", "index.tsv"))

    jobs, scores, speed, log, kidney = [], [], [], [], []
    for run in runs:
        rid = run["run_id"]
        rdir = os.path.join(ROOT, "runs", rid)
        os.makedirs(rdir, exist_ok=True)
        # Small result files come back through the transfer node, while a run is still going. A finished run's tables
        # are final and may have been renamed by role (relabel_tools.py), so they are not fetched again.
        if run.get("host", "").lower().startswith("hpc3") and run.get("status", "").upper() != "DONE" and not VIA:
            for sub in ("out/deliverables/", "out/envs/"):
                subprocess.run(
                    ["rsync", "-a", "--timeout=60", "--exclude=*.h5ad", f"hpc3-xfer:{HPC3_BASE}/{rid}/{sub}", rdir + "/"],
                    capture_output=True, text=True, timeout=300,
                )
        jdefs = read_tsv(os.path.join(rdir, "jobs.tsv"))
        sa = sacct([j["job_id"] for j in jdefs]) or []
        steps = {}
        for r in sa:
            jid = r["JobID"]
            if jid.endswith(".batch"):
                steps[jid[: -len(".batch")]] = r
        for j in jdefs:
            mine = [r for r in sa if "." not in r["JobID"] and r["JobID"].split("_")[0] == j["job_id"]]
            for r in mine:
                for task in expand_tasks(r["JobID"]):
                    key = f"{j['job_id']}_{task}" if task else j["job_id"]
                    what = j.get("unit_names", "").split(",") if j.get("unit_names") else libs
                    unit = what[int(task) - 1] if task.isdigit() and int(task) <= len(what) else j.get("unit", "all")
                    st = r["State"].split(" ")[0]
                    el = num(r["ElapsedRaw"], 0)
                    jobs.append({
                        "job": key,
                        "run_id": rid,
                        "what": j["what"],
                        "tool": j.get("tool", ""),
                        "library": unit,
                        "state": st,
                        "elapsed": hms(el) if st not in ("PENDING",) else "",
                        "elapsed_s": el if st != "PENDING" else None,
                        "node": "" if r["NodeList"].startswith("None") else r["NodeList"],
                        "peak_rss_gb": rss_gb(steps.get(key, {}).get("MaxRSS")),
                        "exit_code": r["ExitCode"] if st not in ("PENDING", "RUNNING") else "",
                        "note": j.get("note", ""),
                    })
        # Scores: metrics table joined with run time.
        met = read_tsv(os.path.join(rdir, "species_mix_metrics.tsv"))
        rt = {(r["dataset"], r["tool"]): r for r in read_tsv(os.path.join(rdir, "runtime.tsv"))}
        raw_pct = {r["dataset"]: num(r["raw_cross_species_pct"], 2) for r in met if r["tool"] == "raw"}
        for r in met:
            if r["tool"] == "raw" or r["tool"].startswith("exp_") or r["tool"] in HIDE_TOOLS:
                continue  # development experiments are scored separately (out/metrics_exp on HPC3)
            t = rt.get((r["dataset"], r["tool"]), {})
            scores.append({
                "library": r["dataset"],
                "tool": r["tool"],
                "run_id": rid,
                "cells_scored": num(r["n_eval_cells"], 0),
                "cells_missing": num(r["n_missing"], 0),
                "raw_cross_pct": raw_pct.get(r["dataset"]),
                "removed_pct": num(r["cross_species_removed"], 1, 100),
                "cell_median_removed_pct": num(r["median_cell_removal"], 1, 100),
                "native_loss_pct": num(r["native_loss"], 2, 100),
                "own_kept_pct": num(r["own_retained"], 2, 100),
                "same_species_ratio": num(r["same_species_ratio"], 2),
                "above_raw": num(r["n_above_raw"], 0),
                "non_integer": num(r["n_noninteger"], 0),
                "rmsle": num(r["contam_rmsle"], 4),
                "kendall": num(r["contam_kendall"], 3),
                "wall_s": num(t.get("wall_s"), 0),
                "peak_rss_gb": num(t.get("peak_rss_gb"), 1),
                "peak_gpu_gb": num(t.get("peak_gpu_mem_gb"), 2),
                "gpu": t.get("gpu", "") or "",
            })
        for r in read_tsv(os.path.join(rdir, "runtime.tsv")):
            if r["dataset"].startswith("ALL"):
                speed.append({"what": r["dataset"].replace("ALL (", "").rstrip(")"), "tool": r["tool"],
                              "run_id": rid, "wall_s": num(r["wall_s"], 0)})
        scored = {(r["dataset"], r["tool"]) for r in met if not r["tool"].startswith("exp_") and r["tool"] not in HIDE_TOOLS}
        per = [r for r in read_tsv(os.path.join(rdir, "runtime.tsv")) if (r["dataset"], r["tool"]) in scored]
        for tool in sorted({r["tool"] for r in per}):
            w = [num(r["wall_s"]) for r in per if r["tool"] == tool and num(r["wall_s"]) is not None]
            if w:
                speed.append({"what": f"{len(w)} libraries, one run each, summed", "tool": tool, "run_id": rid,
                              "wall_s": int(round(sum(w)))})
        # Tier 2 (mouse kidney strain mixture): per-cell truth from genotypes, marker leakage.
        kid = read_tsv(os.path.join(rdir, "kidney_metrics.tsv"))
        for r in kid:
            if r["tool"] == "raw" or r["tool"].startswith("exp_") or r["tool"] in HIDE_TOOLS:
                continue
            t = rt.get((r["dataset"], r["tool"]), {})
            kidney.append({
                "library": r["dataset"], "kind": r.get("kind", ""), "tool": r["tool"], "run_id": rid,
                "cells_scored": num(r["n_eval_cells"], 0), "cells_missing": num(r["n_missing"], 0),
                "truth_pct": num(r["truth_pct"], 1), "removed_pct": num(r["removed_pct"], 1),
                "removed_over_truth": num(r["removed_over_truth"], 2),
                "rmsle": num(r["contam_rmsle"], 3), "kendall": num(r["contam_kendall"], 2),
                "rmsle_cast": num(r["contam_rmsle_cast"], 3), "kendall_cast": num(r["contam_kendall_cast"], 2),
                "marker_removed_pct": num(r["marker_removed_nonpt"], 1, 100), "marker_kept_pct": num(r["marker_kept_pt"], 1, 100),
                "slc34a1_before_pct": num(r["slc34a1_nonpt_before"], 1), "slc34a1_after_pct": num(r["slc34a1_nonpt_after"], 1),
                "marker_log2fc": num(r["marker_log2fc"], 2),
                "above_raw": num(r["n_above_raw"], 0), "non_integer": num(r["n_noninteger"], 0),
                "wall_s": num(t.get("wall_s"), 0), "peak_rss_gb": num(t.get("peak_rss_gb"), 1),
            })
        for r in read_tsv(os.path.join(rdir, "decisions.tsv")):
            log.append({"at": r["at"], "run_id": rid, "entry": r["entry"]})

    # One row per Slurm job: array tasks are summarised, not listed.
    order = ["FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "RUNNING", "PENDING", "COMPLETED"]
    grouped, seen = [], {}
    for j in jobs:
        jid = j["job"].split("_")[0]
        if jid not in seen:
            seen[jid] = {"job": jid, "run_id": j["run_id"], "what": j["what"], "tool": j["tool"], "tasks": 0,
                         "_states": {}, "elapsed_s": None, "peak_rss_gb": None, "note": j["note"]}
            grouped.append(seen[jid])
        g = seen[jid]
        g["tasks"] += 1
        g["_states"][j["state"]] = g["_states"].get(j["state"], 0) + 1
        if j["elapsed_s"] is not None:
            g["elapsed_s"] = max(g["elapsed_s"] or 0, j["elapsed_s"])
        if j["peak_rss_gb"] is not None:
            g["peak_rss_gb"] = max(g["peak_rss_gb"] or 0, j["peak_rss_gb"])
    for g in grouped:
        st = g.pop("_states")
        worst = next(x for x in order if x in st)
        g["state"] = worst if len(st) == 1 else "MIXED" if worst == "COMPLETED" else worst
        g["detail"] = ", ".join(f"{n} {k.lower()}" for k, n in sorted(st.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
        g["longest_task"] = hms(g["elapsed_s"]) if g["elapsed_s"] is not None else ""
    jobs = grouped

    log.sort(key=lambda r: r["at"], reverse=True)
    for i, r in enumerate(log):
        r["n"] = len(log) - i
    tools = read_tsv(os.path.join(ROOT, "config", "tools.tsv"))
    published = read_tsv(os.path.join(ROOT, "config", "published_reference.tsv"))
    for r in published:
        r["removed_pct"] = num(r["removed_pct"], 1)

    # What Janssen et al. published for the same five libraries (their default settings).
    kidney_published = []
    pub = read_tsv(os.path.join(ROOT, "data", "kidney_strainmix", "janssen_benchmark_default.tsv"))
    by = {}
    for r in pub:
        if r["metric"] in ("rmsle", "tau", "lfc") and r["method"] != "raw":
            key = (r["method"], r["param"], r["replicate"])
            by.setdefault(key, {})[r["metric"]] = r["value"]
    label = {"fpr0.01_total25000": "CellBender 0.2.0", "res1_contAuto": "SoupX 1.6.1",
             "resDefault_emptyFalse": "DecontX", "resDefault_emptyTrue": "DecontX + background"}
    for (method, param, rep), v in sorted(by.items()):
        kidney_published.append({"tool": label.get(param, method), "library": "kidney_" + rep, "rmsle": num(v.get("rmsle"), 3),
                                 "kendall": num(v.get("tau"), 2), "marker_log2fc": num(v.get("lfc"), 2),
                                 "source": "Janssen et al. 2023, Genome Biology"})

    # Every library of the benchmark, with where its thresholds landed (no model fitted).
    TIER = {"t1_tuning": "1 cell-line mixtures (tuning)", "t1_holdout": "1 cell-line mixtures (held out)", "t1_nuclei": "1 nuclei mixture",
            "t2_kidney": "2 kidney strain mixture", "t2_xeno": "2 xenograft skin", "t3_rhapsody": "3 BD Rhapsody", "t4_parse": "4 Parse Evercode",
            "t5_notruth": "5 no truth"}
    TRUTH = {"species": "species", "genotype": "genotype (authors)", "tissue_by_well": "tissue per well", "none": "none"}
    thr = {r["name"]: r for r in read_tsv(os.path.join(ROOT, "runs", "20261116-2010_glymph-datasets", "library_thresholds.tsv"))}
    libraries = []
    for r in read_tsv(os.path.join(ROOT, "config", "datasets.tsv")):
        t = thr.get(r["name"], {})
        if r["name"] == "parse_callus":
            status = "unusable: called cells only"
        elif t:
            status = "ready"
        elif r["format"].startswith("starsolo"):
            status = "being aligned from reads"
        else:
            status = "ingesting"
        libraries.append({
            "library": r["name"], "tier": TIER.get(r["tier"], r["tier"]), "platform": r["platform"], "kind": r["kind"],
            "truth": TRUTH.get(r["truth"], r["truth"]), "status": status, "development": r.get("development", ""),
            "barcodes": num(t.get("barcodes"), 0), "cells": num(t.get("confident_cells"), 0), "vendor_cells": num(t.get("vendor_cells"), 0),
            "thresholds_from": t.get("method", ""), "empty_level": num(t.get("empty_level"), 0), "note": r["note"],
        })

    # All-tier benchmark (20261116-2209): means per group and tool, per-library head to head,
    # truth-free scores, resources and scaling. Rounded here to what the page prints.
    # AT0 is the first all-tier run (Glymph 0.3.0) and AT1 the run of 0.3.1. AT is the run of 0.3.2 beside
    # CellBender 0.4.0 on the same libraries: its tables name the tools by role (relabel_tools.py), so
    # "glymph" is 0.3.2, "cellbender" is 0.4.0, and the versions before are scored beside them as
    # glymph_031 and cellbender032. PREV is the earlier Glymph shown beside the current one per library.
    AT0 = os.path.join(ROOT, "runs", "20261116-2209_glymph-alltier-bench")
    AT1 = os.path.join(ROOT, "runs", "20261117-0607_glymph-031-alltier")
    AT = os.path.join(ROOT, "runs", "20261117-1342_glymph-cb040-bench")
    OTHERS = {"scar": "scAR 0.7.0", "soupx": "SoupX 1.6.2", "decontx": "DecontX", "decontx_bg": "DecontX + background"}
    if os.path.exists(os.path.join(AT, "summary.tsv")):
        PREV = "glymph_031"
        LABEL = {"glymph": "Glymph 0.3.2 (budget 0.4%, default)", "glymph_fpr0": "Glymph 0.3.2 (budget 0)", "glymph_fpr02": "Glymph 0.3.2 (budget 0.2%)",
                 "glymph_fpr1": "Glymph 0.3.2 (budget 1%)", "glymph_rep2": "Glymph 0.3.2 (default, second run)", "glymph_031": "Glymph 0.3.1 (budget 0.4%)",
                 "cellbender": "CellBender 0.4.0 (budget 1%, default)", "cellbender_fpr05": "CellBender 0.4.0 (budget 0.5%)", "cellbender_fpr0": "CellBender 0.4.0 (budget 0)",
                 "cellbender032": "CellBender 0.3.2 (budget 1%, default)", "cellbender032_fpr05": "CellBender 0.3.2 (budget 0.5%)",
                 "cellbender032_fpr0": "CellBender 0.3.2 (budget 0)", **OTHERS}
        RES_TOOLS = ("glymph", "glymph_031", "cellbender", "cellbender032", "scar", "soupx", "decontx", "decontx_bg")
    elif os.path.exists(os.path.join(AT1, "summary.tsv")):
        AT, PREV = AT1, "glymph_030"
        LABEL = {"glymph": "Glymph 0.3.1 (budget 0.4%, default)", "glymph_fpr0": "Glymph 0.3.1 (budget 0)", "glymph_fpr02": "Glymph 0.3.1 (budget 0.2%)",
                 "glymph_fpr1": "Glymph 0.3.1 (budget 1%)", "glymph_030": "Glymph 0.3.0 (budget 0.4%)", "glymph_030_fpr0": "Glymph 0.3.0 (budget 0)",
                 "glymph_030_fpr1": "Glymph 0.3.0 (budget 1%)", "cellbender": "CellBender 0.3.2 (budget 1%, default)",
                 "cellbender_fpr05": "CellBender 0.3.2 (budget 0.5%)", "cellbender_fpr0": "CellBender 0.3.2 (budget 0)", **OTHERS}
        RES_TOOLS = ("glymph", "glymph_030", "cellbender", "scar", "soupx", "decontx", "decontx_bg")
    else:
        AT, PREV = AT0, None
        LABEL = {"glymph": "Glymph 0.3.0 (budget 0.4%, default)", "glymph_fpr0": "Glymph 0.3.0 (budget 0)", "glymph_fpr02": "Glymph 0.3.0 (budget 0.2%)",
                 "glymph_fpr1": "Glymph 0.3.0 (budget 1%)", "cellbender": "CellBender 0.3.2 (budget 1%, default)",
                 "cellbender_fpr05": "CellBender 0.3.2 (budget 0.5%)", "cellbender_fpr0": "CellBender 0.3.2 (budget 0)", **OTHERS}
        RES_TOOLS = ("glymph", "cellbender", "scar", "soupx", "decontx", "decontx_bg")
    rank = {t: i for i, t in enumerate(LABEL)}
    GORDER = ["cell-line mixtures, other (10x)", "BD Rhapsody", "Parse Evercode", "xenograft skin (10x)", "nuclei mixture (10x)",
              "cell-line mixtures, tuning (10x)", "kidney strain mixture (10x)", "cortex nuclei (10x)"]
    grank = {g: i for i, g in enumerate(GORDER)}
    alltier, alltier_kidney, alltier_free = [], [], []
    for r in sorted(read_tsv(os.path.join(AT, "summary.tsv")), key=lambda r: (grank.get(r["group"], 99), rank.get(r["tool"], 99))):
        if r["tool"] not in LABEL:
            continue
        base = {"group": r["group"], "tool": LABEL[r["tool"]], "libraries": num(r["n"], 0)}
        if r["scores"] == "species":
            alltier.append({**base, "scope": r["libraries"], "removed_pct": num(r["cross_species_removed"], 1, 100),
                            "own_kept_pct": num(r["own_retained"], 2, 100), "native_loss_pct": num(r["native_loss"], 2, 100),
                            "same_species_ratio": num(r["same_species_ratio"], 2), "rmsle": num(r["contam_rmsle"], 3),
                            "kendall": num(r["contam_kendall"], 2),
                            "adds_counts": "yes" if (num(r["n_above_raw"]) or 0) > 0 else "no",
                            "whole_numbers": "no" if (num(r["n_noninteger"]) or 0) > 0 else "yes"})
        elif r["scores"] == "kidney":
            alltier_kidney.append({"tool": LABEL[r["tool"]], "libraries": num(r["n"], 0), "rmsle": num(r["contam_rmsle"], 3),
                                   "kendall": num(r["contam_kendall"], 3), "removed_over_truth": num(r["removed_over_truth"], 2),
                                   "marker_removed_pct": num(r["marker_removed_nonpt"], 1, 100), "marker_kept_pct": num(r["marker_kept_pt"], 1, 100)})
        elif r["scores"] == "generic" and r["libraries"] == "all":
            alltier_free.append({**base, "removed_pct": num(r["removed_pct"], 1), "marker_removed_pct": num(r["marker_removed_off"], 1, 100),
                                 "marker_kept_pct": num(r["marker_kept_on"], 1, 100)})
    sp_rows = read_tsv(os.path.join(AT, "scores_species.tsv"))
    by_lib = {}
    for r in sp_rows:
        by_lib.setdefault(r["dataset"], {})[r["tool"]] = r
    alltier_lib = []
    for lib, t in by_lib.items():
        g, c = t.get("glymph"), t.get("cellbender")
        if not g or not c:
            continue
        g0 = t.get(PREV, {}) if PREV else {}
        alltier_lib.append({
            "library": lib, "platform": g.get("platform", ""), "development": g.get("development", ""),
            "glymph030_removed": num(g0.get("cross_species_removed"), 1, 100), "glymph030_rmsle": num(g0.get("contam_rmsle"), 3),
            "cross_pct": num(t["raw"]["raw_cross_species_pct"], 2) if "raw" in t else None, "masked_pct": num(g.get("masked_cross_pct"), 1),
            "glymph_removed": num(g["cross_species_removed"], 1, 100), "cellbender_removed": num(c["cross_species_removed"], 1, 100),
            "glymph_kept": num(g["own_retained"], 2, 100), "cellbender_kept": num(c["own_retained"], 2, 100),
            "glymph_rmsle": num(g["contam_rmsle"], 3), "cellbender_rmsle": num(c["contam_rmsle"], 3),
            "glymph_kendall": num(g["contam_kendall"], 2), "cellbender_kendall": num(c["contam_kendall"], 2)})
    alltier_lib.sort(key=lambda r: (r["platform"], r["library"]))
    res = [r for r in read_tsv(os.path.join(AT, "resources.tsv")) if r["tool"] in RES_TOOLS]
    alltier_res = []

    def med(v):
        v = sorted(v)
        return None if not v else (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))

    for tool in RES_TOOLS:
        rows = [r for r in res if r["tool"] == tool]
        w = [num(r["wall_s"]) for r in rows if num(r["wall_s"]) is not None]
        m = [num(r["peak_rss_gb"]) for r in rows if num(r["peak_rss_gb"]) is not None]
        gm = [num(r["gpu_peak_mem_gb"]) for r in rows if num(r.get("gpu_peak_mem_gb")) is not None]
        if not w:
            continue
        alltier_res.append({"what": f"{len(w)} libraries, one run each", "tool": LABEL[tool].split(" (")[0], "hardware": "one A30" if gm else "4 CPU cores",
                            "median_wall_s": int(round(med(w))), "longest_wall_s": int(round(max(w))), "total_hours": round(sum(w) / 3600, 1),
                            "median_ram_gb": round(med(m), 1) if m else None, "peak_ram_gb": round(max(m), 1) if m else None,
                            "peak_gpu_gb": round(max(gm), 1) if gm else None})
    for r in read_tsv(os.path.join(AT if os.path.exists(os.path.join(AT, "resources_extra.tsv")) else AT0, "resources_extra.tsv")):
        alltier_res.append({"what": r["what"], "tool": r["tool"], "hardware": r["hardware"], "median_wall_s": num(r["wall_s"], 0), "longest_wall_s": None,
                            "total_hours": None, "median_ram_gb": None, "peak_ram_gb": num(r["peak_rss_gb"], 1), "peak_gpu_gb": num(r["peak_gpu_gb"], 1)})
    alltier_scaling = [{"tool": r["tool"], "cells": num(r["cells"], 0), "state": r["state"], "wall_s": num(r["wall_s"], 0),
                        "peak_ram_gb": num(r["peak_rss_gb"], 1), "note": r["note"]} for r in read_tsv(os.path.join(AT if os.path.exists(os.path.join(AT, "scaling.tsv")) else AT0, "scaling.tsv"))]
    artifacts = [{"library": r["dataset"], "gene": r["gene"], "in_cells_of": r["in_cells_of"], "share_of_cross_pct": num(r["share_of_cross_pct"], 1),
                  "share_in_source_cells_pct": num(r["share_in_source_cells_pct"], 2), "share_in_soup_pct": num(r["share_in_soup_pct"], 2)}
                 for r in read_tsv(os.path.join(AT, "artifact_genes.tsv"))]

    tracker = {
        "generated_at": now,
        "alltier": alltier,
        "alltier_lib": alltier_lib,
        "alltier_kidney": alltier_kidney,
        "alltier_free": alltier_free,
        "alltier_res": alltier_res,
        "alltier_scaling": alltier_scaling,
        "alltier_artifacts": artifacts,
        "libraries": libraries,
        "runs": runs,
        "jobs": jobs,
        "scores": scores,
        "speed": speed,
        "tools": tools,
        "published": published,
        "kidney": kidney,
        "kidney_published": kidney_published,
        "log": log,
    }
    os.makedirs(os.path.join(ROOT, "dashboard"), exist_ok=True)
    out = os.path.join(ROOT, "dashboard", "tracker.json")
    with open(out, "w") as f:
        json.dump(tracker, f, indent=1)

    # runs/INDEX.md is generated from runs/index.tsv so the two cannot drift.
    with open(os.path.join(ROOT, "runs", "INDEX.md"), "w") as f:
        f.write("# Runs\n\nGenerated from `runs/index.tsv` by `code/dashboard/build_tracker.py`. "
                "Edit the TSV, not this file.\n\n")
        f.write("run_id | host | job id or tmux | status | summary\n---|---|---|---|---\n")
        for r in runs:
            f.write(f"{r['run_id']} | {r['host']} | {r['jobs']} | {r['status']} | {r['summary']}\n")

    states = {}
    for j in jobs:
        states[j["state"]] = states.get(j["state"], 0) + 1
    print(f"[tracker] {out}: {len(jobs)} jobs {states}, {len(scores)} score rows, {len(log)} log entries")


if __name__ == "__main__":
    main()
