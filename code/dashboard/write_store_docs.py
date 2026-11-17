#!/usr/bin/env python3
"""Write the documents the dashboard's store holds, as JSON files under dashboard/store/.

    python3 code/dashboard/write_store_docs.py <blob id of the uploaded tracker.json>

The files are then written to the dashboard with one batch call. The page itself is
dashboard/index.html; every table on it is read from the one tracker file.
"""

import datetime as dt
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
blob = sys.argv[1]
out = os.path.join(ROOT, "dashboard", "store")
os.makedirs(out, exist_ok=True)
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

DATASETS = {
    "runs": ("Benchmark runs", "One row per run, from the project's run index."),
    "libraries": ("Benchmark libraries", "Every library of the benchmark: tier, platform, kind of truth, whether it is ready, and the cell count from the depth distribution beside the vendor's."),
    "jobs": ("Slurm jobs on HPC3", "One row per Slurm job, array tasks summarised, with state and longest task time from the scheduler's accounting."),
    "scores": ("Scores by library and tool", "Species-mixing metrics joined with run time, one row per library and tool."),
    "speed": ("Whole-benchmark wall time", "Wall time summed over the libraries, and of the run on one 4-GPU node."),
    "tools": ("Tools and environments", "Each tool, its version and the HPC3 environment it runs in."),
    "published": ("Published reference values", "Cross-species removal quoted from Schroeder et al. 2026 (bioRxiv). Not rerun here."),
    "log": ("Decision log", "Dated decisions and events of the benchmark, written as they happened."),
    "alltier": ("All tiers, species truth by group", "Means per group of libraries and tool on the human + mouse mixtures of every platform, alignment-artifact genes masked; once over the held-out libraries and once over all."),
    "alltier_lib": ("All tiers, Glymph against CellBender by library", "One row per species-truth library: both tools at their defaults."),
    "alltier_kidney": ("All tiers, mouse kidney", "Means over the five kidney strain-mixture libraries, authors' genotype truth."),
    "alltier_free": ("All tiers, scores without truth", "Share of counts removed and leakage of cluster-exclusive marker genes, per group and tool, every library."),
    "alltier_res": ("All tiers, time and memory", "Wall time and memory per tool over the 54 libraries, and the special runs: one 4-GPU node, CPU only, other GPU models."),
    "alltier_scaling": ("Scaling with library size", "One library copied 1 to 8 times: wall time, memory and whether the run finished."),
    "alltier_artifacts": ("Genes masked from the species score", "Genes whose cross-species counts are alignment artifacts or transgenes, found from raw counts alone."),
    "kidney": ("Tier 2 scores, mouse kidney", "One row per kidney library and tool: per-cell contamination against the authors' genotype truth, and marker leakage."),
    "kidney_published": ("Tier 2 published reference", "Per-cell error, rank agreement and marker fold change that Janssen et al. 2023 reported for the same libraries. Not rerun here."),
}
for key, (title, desc) in DATASETS.items():
    doc = {
        "title": title,
        "description": desc,
        "source": {"kind": "file", "url": f"/_blob/{blob}", "name": "tracker.json", "format": "json"},
        "updated": {"at": now, "by": "Claude, from HPC3 job accounting and result files"},
        "fixed": True,
    }
    json.dump(doc, open(os.path.join(out, f"dataset_{key}.json"), "w"), indent=1)
json.dump({"title": "Glymph benchmark tracker", "lang": "en"}, open(os.path.join(out, "meta.json"), "w"))
json.dump({"text": open(os.path.join(ROOT, "dashboard", "index.html")).read()},
          open(os.path.join(out, "index_html.json"), "w"))
print("wrote", sorted(os.listdir(out)), now)
