#!/usr/bin/env python
"""Collect benchmark tables and draw the summary figure.

Reads   <run>/out/metrics/*.tsv, <run>/out/<tool>/*.time.txt, <run>/out/glymph/*.report.json
Writes  species_mix_metrics.tsv, runtime.tsv, benchmark_summary.pdf into --out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

# Three groups, validated with the dataviz palette checker (light surface): the method under
# test, the baseline to beat, every other tool. Tool names sit on the axis, so identity never
# rests on colour alone.
GROUP_COLORS = {"Glymph": "#C2872B", "CellBender": "#3A6FB5", "Other tools": "#2F9B70"}
INK, MUTED, GRID = "#16233A", "#6E7887", "#E4E7EC"
TOOL_NAMES = {
    "glymph": "Glymph 0.2.0 (default)", "glymph_fpr0": "Glymph 0.2.0, no false-positive budget",
    "glymph_fpr1": "Glymph 0.2.0, false-positive budget 1%", "glymph_v01": "Glymph 0.1.0", "glymph_v0": "Glymph 0.0.1",
    "cellbender": "CellBender 0.3.2", "scar": "scAR 0.7.0", "soupx": "SoupX 1.6.2", "decontx": "DecontX 1.8.0",
    "decontx_bg": "DecontX 1.8.0 + background", "ambidose": "AmbiDose 0.5.15",
}


def tool_group(tool):
    return "Glymph" if tool.startswith("glymph") else "CellBender" if tool == "cellbender" else "Other tools"


def dot_rows_figure(d, panels, path, note, sort_col, ascending=True):
    """One row per tool, one panel per score. Small dots are libraries, the large dot is their mean.

    d: one row per (dataset, tool) with the score columns. panels: (column, title, hint, number format, log scale).
    Tools are ordered by the mean of sort_col, best at the top.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tools = d.groupby("tool")[sort_col].mean().sort_values(ascending=ascending).index.tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.75 * len(panels), 0.46 * len(tools) + 2.0), sharey=True)
    for ax, (col, title, hint, fmt, log) in zip(axes, panels):
        for i, tool in enumerate(tools):
            v = pd.to_numeric(d.loc[d.tool == tool, col], errors="coerce").dropna()
            if v.empty:
                continue
            color = GROUP_COLORS[tool_group(tool)]
            ax.scatter(v, np.full(len(v), i), s=16, color=color, alpha=0.45, linewidths=0, zorder=2)
            ax.scatter([v.mean()], [i], s=80, color=color, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(fmt.format(v.mean()), (1.0, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=8, color=INK)
        if log:
            ax.set_xscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=16)
        ax.text(0.0, 1.02, hint, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    n_lib = d.dataset.nunique()
    n_by = d.groupby("tool").dataset.nunique()
    labels = [TOOL_NAMES.get(t, t) + (f" ({n_by[t]} of {n_lib} libraries)" if n_by[t] < n_lib else "") for t in tools]
    axes[0].set_yticks(range(len(tools)))
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    axes[0].set_ylim(-0.6, len(tools) - 0.4)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, markeredgecolor="white", label=g)
               for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False, fontsize=9)
    fig.text(0.99, 0.012, f"Small dots: the {n_lib} libraries. Large dot and number: their mean. " + note,
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.23, right=0.965, top=0.86, bottom=0.14, wspace=0.28)
    fig.savefig(path)
    plt.close(fig)


def summary_figure(met, rt, path):
    """Tier 1 (species mixtures): removal, retention, per-cell error, wall time."""
    import numpy as np

    d = met[met.tool != "raw"].copy()
    if not rt.empty:
        d = d.merge(rt[["dataset", "tool", "wall_s"]], on=["dataset", "tool"], how="left")
    else:
        d["wall_s"] = np.nan
    d["removed"] = 100 * d.cross_species_removed
    d["kept"] = 100 * d.own_retained
    panels = [
        ("removed", "Cross-species counts removed, %", "right is better", "{:.1f}", False),
        ("kept", "Same-species counts kept, %", "right is better", "{:.2f}", False),
        ("contam_rmsle", "Per-cell contamination error", "left is better", "{:.4f}", False),
        ("wall_s", "Wall time per library, s", "left is better, log scale", "{:.0f}", True),
    ]
    dot_rows_figure(d, panels, path, "scAR writes entries above the raw counts; SoupX and DecontX write fractional counts.",
                    sort_col="removed")


def runtime_table(run_dir, met):
    """Wall time and peak memory of every scored (library, tool) pair, plus the multi-GPU batch."""
    rows = []
    tools = sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(run_dir, "out/*/*.time.txt"))})
    tools = [t for t in tools if not t.endswith("_failed") and not t.startswith("exp_") and not t.startswith("glymph_4gpu")]
    scored = set(zip(met["dataset"], met["tool"])) if not met.empty else set()
    for tool in tools:
        for f in sorted(glob.glob(os.path.join(run_dir, f"out/{tool}/*.time.txt"))):
            name = os.path.basename(f)[: -len(".time.txt")]
            if (name, tool) not in scored:  # a timing file without a scored output is a failed or unfinished run
                continue
            row = {"dataset": name, "tool": tool, "wall_s": None, "peak_rss_gb": None}
            row.update(parse_profile(f[: -len(".time.txt")]))
            rep = os.path.join(run_dir, f"out/{tool}/{name}.report.json")
            if tool.startswith("glymph") and os.path.exists(rep):
                r = json.load(open(rep))
                row.update({f"glymph_{k}": v for k, v in r["timings"].items()})
                row.update({"gpu": r["gpu"], "peak_gpu_mem_gb": r["peak_gpu_mem_gb"], "n_cells": r["n_cells"],
                            "epochs_run": r["train"]["epochs_run"]})
            rows.append(row)
    summ = os.path.join(run_dir, "out/glymph_4gpu/batch_summary.json")
    if os.path.exists(summ):
        s = json.load(open(summ))
        if s.get("n_done") == s["n_libraries"]:  # only a batch in which every library finished counts
            rows.append({"dataset": f"ALL ({s['n_libraries']} libraries, {len(s['devices'])} GPUs)", "tool": "glymph",
                         "wall_s": s["wall_s"]})
    return pd.DataFrame(rows)


def parse_time(path):
    """Wall seconds and peak RSS (GB) from a `/usr/bin/time -v` file."""
    d = parse_profile(path[: -len(".time.txt")])
    return d.get("wall_s"), d.get("peak_rss_gb")


def parse_profile(prefix):
    """Everything profile.sh recorded for one run: time, CPU, memory, GPU."""
    out = {}
    try:
        txt = open(prefix + ".time.txt").read()
    except OSError:
        return out
    wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", txt)
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        out["wall_s"] = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    for key, pat, scale in (("peak_rss_gb", r"Maximum resident set size \(kbytes\): (\d+)", 1e-6),
                            ("cpu_user_s", r"User time \(seconds\): ([\d.]+)", 1.0),
                            ("cpu_sys_s", r"System time \(seconds\): ([\d.]+)", 1.0),
                            ("cpu_pct", r"Percent of CPU this job got: (\d+)%", 1.0)):
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1)) * scale
    if "cpu_user_s" in out and "cpu_sys_s" in out:
        out["cpu_core_s"] = out["cpu_user_s"] + out["cpu_sys_s"]
    if os.path.exists(prefix + ".host.txt"):
        for line in open(prefix + ".host.txt"):
            k, _, v = line.strip().partition(" ")
            if k in ("node", "cores_granted", "gpu"):
                out["gpu_model" if k == "gpu" else k] = v
    if os.path.exists(prefix + ".gpu.csv"):
        try:
            g = pd.read_csv(prefix + ".gpu.csv", header=None, names=["t", "index", "util", "mem"], skipinitialspace=True)
            g["util"], g["mem"] = pd.to_numeric(g.util, errors="coerce"), pd.to_numeric(g.mem, errors="coerce")
            n_gpu = g["index"].nunique()
            base = g.groupby("index").mem.first()  # reading before the command started
            run = g.iloc[n_gpu:]
            if len(run):
                peak = (run.groupby("index").mem.max() - base).clip(lower=0)
                out["gpu_peak_mem_gb"] = float(peak.max()) / 1024.0
                out["gpu_mean_util_pct"] = float(run.util.mean())
                out["gpu_mem_before_gb"] = float(base.max()) / 1024.0
        except Exception:  # noqa: BLE001  (a truncated sample file must not stop the collection)
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.run_dir, "out/metrics/*.tsv")))
    met = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True) if files else pd.DataFrame()
    met.to_csv(os.path.join(a.out, "species_mix_metrics.tsv"), sep="\t", index=False)

    rt = runtime_table(a.run_dir, met)
    rt.to_csv(os.path.join(a.out, "runtime.tsv"), sep="\t", index=False)

    if met.empty:
        print("no metrics found")
        return
    summary_figure(met, rt, os.path.join(a.out, "benchmark_summary.pdf"))
    print(met.to_string(index=False))
    print(rt.to_string(index=False))


if __name__ == "__main__":
    main()
