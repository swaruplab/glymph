#!/usr/bin/env python3
"""Figures of the all-tier benchmark, from the tables in a run folder (MacBook, matplotlib).

    python3 code/benchmark/plot_all_tiers.py runs/20261116-2209_glymph-alltier-bench [Glymph version, default 0.3.0]
        [CellBender tool folder, default cellbender] [CellBender version, default 0.3.2]
    python3 code/benchmark/plot_all_tiers.py runs/20261117-1342_glymph-cb040-bench 0.3.2 cellbender040 0.4.0

  figures/tradeoff_by_group.{png,pdf}   removal against same-species counts kept, one panel per group;
                                        Glymph and CellBender as lines over their false-positive budgets
  figures/percell_error.{png,pdf}       per-cell error, Glymph against CellBender, one point per library
  figures/time_and_scaling.{png,pdf}    wall time per library by tool, and time against library size
Colours are the validated five-slot palette (dataviz skill); every series also has its own marker
and a direct label, since three of the five colours are below 3:1 contrast on the surface.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

run = sys.argv[1]
VER = sys.argv[2] if len(sys.argv) > 2 else "0.3.0"  # Glymph version scored as the tool "glymph" in this run
CB = sys.argv[3] if len(sys.argv) > 3 else "cellbender"  # the CellBender shown: its tool folder (budgets in <CB>_fpr0, <CB>_fpr05)
CBV = sys.argv[4] if len(sys.argv) > 4 else "0.3.2"
out = os.path.join(run, "figures")
os.makedirs(out, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Glymph": "#2a78d6", "CellBender": "#eb6834", "scAR": "#1baf7a", "SoupX": "#eda100", "DecontX + bg": "#e87ba4"}
MARK = {"Glymph": "o", "CellBender": "s", "scAR": "^", "SoupX": "D", "DecontX + bg": "v"}
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".png"))


s = pd.read_csv(os.path.join(run, "summary.tsv"), sep="\t")

# ---- 1. trade-off by group
want = [("held out", "cell-line mixtures, other (10x)", "10x cell lines, held out"), ("held out", "BD Rhapsody", "BD Rhapsody, held out"),
        ("held out", "xenograft skin (10x)", "Xenograft skin, held out"), ("all", "Parse Evercode", "Parse Evercode, all"),
        ("all", "nuclei mixture (10x)", "Nuclei mixture"), ("all", "cell-line mixtures, tuning (10x)", "10x cell lines, tuning")]
panels = []
for scope, group, title in want:
    sel = s[(s.scores == "species") & (s.libraries == scope) & (s.group == group) & (s.tool == "glymph")]
    if len(sel):
        panels.append((scope, group, f"{title} ({int(sel.n.iloc[0])})"))
fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
for ax, (scope, group, title) in zip(axes.ravel(), panels):
    t = s[(s.scores == "species") & (s.libraries == scope) & (s.group == group)].set_index("tool")
    xs = []
    for name, tools, budgets, side in (("Glymph", ["glymph_fpr0", "glymph_fpr02", "glymph", "glymph_fpr1"], ["0", "0.2", "0.4", "1"], -1),
                                       ("CellBender", [f"{CB}_fpr0", f"{CB}_fpr05", CB], ["0", "0.5", "1"], 1)):
        x, y = 100 * t.loc[tools, "own_retained"], 100 * t.loc[tools, "cross_species_removed"]
        xs += list(x)
        ax.plot(x, y, color=COL[name], lw=2, marker=MARK[name], ms=5.5, mec=SURF, mew=1, zorder=3, label=name)
        for k in (0, len(tools) - 1):  # only the smallest and the largest budget are labelled
            ax.annotate(budgets[k], (x.iloc[k], y.iloc[k]), xytext=(8 * side, 0), textcoords="offset points",
                        ha="left" if side > 0 else "right", va="center", fontsize=7.5, color=INK2)
    others = [(n, tl) for n, tl in (("SoupX", "soupx"), ("DecontX + bg", "decontx_bg"), ("scAR", "scar")) if tl in t.index]
    # The x axis follows Glymph, CellBender, SoupX and DecontX; a tool far off it is named at the edge with its values.
    xs += [100 * t.loc[tl, "own_retained"] for n, tl in others if n != "scAR"]
    lo = min(xs) - 0.25 * (max(xs) - min(xs)) - 0.05
    hi = max(xs) + 0.12 * (max(xs) - min(xs)) + 0.05
    off = []
    for name, tool in others:
        x, y = 100 * t.loc[tool, "own_retained"], 100 * t.loc[tool, "cross_species_removed"]
        if x < lo:
            off.append(f"{name}: kept {x:.1f}%, removed {y:.1f}% (off the axis)")
            continue
        ax.plot([x], [y], ls="", marker=MARK[name], ms=6, color=COL[name], mec=SURF, mew=1, zorder=2, label=name)
    ax.set_xlim(lo, hi)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.margins(y=0.14)
    if off:
        ax.text(0.02, 0.03, "\n".join(off), transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")
handles = {}
for ax in axes.ravel():
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(l, h)
order = [n for n in ("Glymph", "CellBender", "scAR", "SoupX", "DecontX + bg") if n in handles]
fig.legend([handles[n] for n in order], [f"Glymph {VER}" if n == "Glymph" else f"CellBender {CBV}" if n == "CellBender" else n for n in order],
           loc="upper right", ncol=len(order), frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 0.965))
for ax in axes[-1]:
    ax.set_xlabel("Same-species counts kept, %")
for ax in axes[:, 0]:
    ax.set_ylabel("Cross-species counts removed, %")
fig.suptitle("Removal against counts kept, along each tool's false-positive budget (end points labelled, in %)", x=0.01, ha="left", fontsize=11, color=INK)
fig.text(0.01, -0.01, "Up and to the right is better, but same-species counts also hold background, so keeping 100% is not the aim. "
         "Alignment-artifact genes masked for every tool. scAR also adds counts; SoupX and DecontX return fractions.", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.01, 1, 0.94))
save(fig, "tradeoff_by_group")

# ---- 2. per-cell error, library by library
sp = pd.read_csv(os.path.join(run, "scores_species.tsv"), sep="\t")
p = sp[sp.tool.isin(["glymph", CB])].pivot(index="dataset", columns="tool", values="contam_rmsle").rename(columns={CB: "cellbender"})
meta = sp.drop_duplicates("dataset").set_index("dataset")
kid = pd.read_csv(os.path.join(run, "scores_kidney.tsv"), sep="\t")
pk = kid[kid.tool.isin(["glymph", CB])].pivot(index="dataset", columns="tool", values="contam_rmsle").rename(columns={CB: "cellbender"})
p = pd.concat([p, pk]).sort_values("cellbender")
dev = pd.concat([meta.development, kid.drop_duplicates("dataset").set_index("dataset").development])
fig, ax = plt.subplots(figsize=(7.2, 7.4))
yy = range(len(p))
ax.hlines(yy, p.glymph, p.cellbender, color=GRID, lw=2, zorder=1)
ax.plot(p.cellbender, yy, ls="", marker=MARK["CellBender"], ms=6, color=COL["CellBender"], mec=SURF, mew=1, label=f"CellBender {CBV}", zorder=3)
ax.plot(p.glymph, yy, ls="", marker=MARK["Glymph"], ms=6, color=COL["Glymph"], mec=SURF, mew=1, label=f"Glymph {VER}", zorder=3)
ax.set_yticks(list(yy))
ax.set_yticklabels([f"{n}{'' if dev[n] == 'held out' else '  (' + dev[n] + ')'}" for n in p.index], fontsize=7.5)
ax.set_xlabel("Per-cell error of the contamination fraction (RMSLE, lower is better)")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
wins = int((p.glymph < p.cellbender).sum())
ax.set_title(f"Per-cell error: Glymph {VER} lower on {wins} of {len(p)} libraries with truth", loc="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "percell_error")

# ---- 3. time per library, and scaling
r = pd.read_csv(os.path.join(run, "resources.tsv"), sep="\t")
names = {"glymph": "Glymph", CB: "CellBender", "scar": "scAR", "soupx": "SoupX", "decontx_bg": "DecontX + bg"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
for i, (tool, name) in enumerate(names.items()):
    w = r[r.tool == tool].wall_s.sort_values().to_numpy()
    jitter = [(j % 7 - 3) * 0.045 for j in range(len(w))]
    a1.plot([i + d for d in jitter], w, ls="", marker=MARK[name], ms=3.8, color=COL[name], alpha=0.75, mec="none")
    med = float(pd.Series(w).median())
    a1.hlines(med, i - 0.3, i + 0.3, color=INK, lw=1.6, zorder=4)
    a1.annotate(f"{med:,.0f} s", (i + 0.32, med), xytext=(2, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
a1.set_yscale("log")
a1.set_xticks(range(len(names)))
a1.set_xticklabels([f"{n}\n({'A30' if t in ('glymph', CB, 'scar') else '4 CPU cores'})" for t, n in names.items()])
a1.set_ylabel("Wall time per library, s (log scale)")
a1.grid(axis="x", visible=False)
a1.set_title(f"Wall time, 54 libraries, Glymph {VER} (bar = median)", loc="left", fontsize=10.5, color=INK)
sc_path = os.path.join(run, "scaling.tsv")
if not os.path.exists(sc_path):
    sc_path = os.path.join(os.path.dirname(os.path.abspath(run)), "20261116-2209_glymph-alltier-bench", "scaling.tsv")
sc = pd.read_csv(sc_path, sep="\t")
exact = {"Glymph": f"Glymph {VER}", "CellBender": f"CellBender {CBV}"}
if not all((sc.tool == v).any() for v in exact.values()):  # tables written before versions were listed side by side
    later = sc[sc.tool.str.contains("0.3.1")]
    if len(later) and VER != "0.3.1":  # the memory fix, run after the benchmark: dashed, same colour as Glymph
        a2.plot(later.cells / 1000, later.wall_s / 60, color=COL["Glymph"], lw=1.6, ls=(0, (4, 2)), marker=MARK["Glymph"], ms=5, mfc=SURF, mec=COL["Glymph"], mew=1.4)
        a2.annotate("Glymph 0.3.1, in development", (later.cells.iloc[-1] / 1000, later.wall_s.iloc[-1] / 60), xytext=(0, 8), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
        sc = sc[~sc.tool.str.contains("0.3.1")]
    exact = None
for name in ("Glymph", "CellBender"):
    t = sc[sc.tool == exact[name]] if exact else sc[sc.tool.str.startswith(name)]
    ok = t[t.state == "completed"]
    a2.plot(ok.cells / 1000, ok.wall_s / 60, color=COL[name], lw=2, marker=MARK[name], ms=6, mec=SURF, mew=1)
    a2.annotate(f"{t.tool.iloc[0]}", (ok.cells.iloc[-1] / 1000, ok.wall_s.iloc[-1] / 60), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK, weight="bold")
    bad = t[t.state != "completed"]
    if len(bad):
        ybad = ok.wall_s.iloc[0] / 60 * 0.9
        a2.plot(bad.cells / 1000, [ybad] * len(bad), ls="", marker="x", ms=8, mew=2, color=COL[name])
        for c in bad.cells:
            a2.annotate("out of GPU memory", (c / 1000, ybad), xytext=(0, 8), textcoords="offset points", ha="right" if c > 100000 else "center", fontsize=7.5, color=INK2)
a2.set_yscale("log")
a2.set_xlabel("Cells in the library, thousands (a 20,000-cell library copied 1 to 8 times)")
a2.set_ylabel("Wall time, min (log scale)")
a2.set_title("Time against library size, one A30", loc="left", fontsize=10.5, color=INK)
fig.tight_layout()
save(fig, "time_and_scaling")
