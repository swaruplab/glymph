#!/usr/bin/env python3
"""Benchmark figures for the Glymph deck and website, rendered per theme with a transparent background.

    python3 code/deck/make_deck_figures.py dark [run folder of the all-tier tables]

Writes reports/deck_figures/<theme>/*.png and reports/deck_figures/numbers.json (every number the slides quote,
computed here from the tables so the deck cannot drift from them). Colours are the deck's tokens: Glymph in the
validated warm data colour, CellBender in the cool one, every other tool in muted grey with a direct label.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

THEME = sys.argv[1] if len(sys.argv) > 1 else "dark"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "runs", "20261117-1342_glymph-cb040-bench")
# Which versions the tools "glymph" and "cellbender" of this run's tables are (versions.json in the run folder).
VERS = json.load(open(os.path.join(RUN, "versions.json"))) if os.path.exists(os.path.join(RUN, "versions.json")) else {"glymph": "0.3.1", "cellbender": "0.3.2"}
GV, CV = VERS["glymph"], VERS["cellbender"]
RUN0 = os.path.join(ROOT, "runs", "20261116-2209_glymph-alltier-bench")
OUT = os.path.join(ROOT, "reports", "deck_figures", THEME)
os.makedirs(OUT, exist_ok=True)
T = {"light": dict(ink="#16233A", ink2="#3B4658", muted="#6E7887", hair="#E4E7EC", warm="#C2872B", cool="#3A6FB5", surface="#FFFFFF"),
     "dark": dict(ink="#F3EEE4", ink2="#C5CBD6", muted="#8C96A8", hair="#27324A", warm="#BF8536", cool="#5B93E6", surface="#0E1524")}[THEME]
plt.rcParams.update({"font.family": "Avenir Next", "font.size": 13, "axes.edgecolor": T["hair"], "axes.labelcolor": T["ink2"],
                     "xtick.color": T["ink2"], "ytick.color": T["ink2"], "text.color": T["ink"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": T["hair"], "grid.linewidth": 0.7,
                     "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none", "axes.titlesize": 14,
                     "axes.titlecolor": T["ink"], "axes.labelsize": 13, "legend.frameon": False})
G, B = "Glymph", "CellBender"
NUM = {"glymph_version": GV, "cellbender_version": CV}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


def table(name, run=RUN):
    p = os.path.join(run, name)
    if not os.path.exists(p):
        p = os.path.join(run, "first_pass", name)
    return pd.read_csv(p, sep="\t")


summ, sp, kid, res = table("summary.tsv"), table("scores_species.tsv"), table("scores_kidney.tsv"), table("resources.tsv")
CONF = ["hgmm_20k_5pv2_HT", "hgmm_10k_5p_gemx", "hgmm_10k_3p_chromX", "hgmm_5k_3p_multiplex", "hgmm_1k_3p_LT", "hgmm_10k_v3_DI", "hgmm_1k_v2",
        "xeno_noHSC_1", "xeno_noHSC_2", "xeno_noHSC_3", "xeno_HSC_2", "bd_hgmm_mix2"]
GROUPS = {"10x cell lines": [d for d in sp.dataset.unique() if d.startswith(("hgmm", "jurkat"))],
          "BD Rhapsody": ["bd_hgmm_mix1", "bd_hgmm_mix2"], "Parse": ["parse_hgmm_sub1", "parse_hgmm_sub2"],
          "Xenograft": [d for d in sp.dataset.unique() if d.startswith("xeno")], "Nuclei mix": ["neat_barnyard"]}


def mean(tool, col, libs, t=sp):
    v = t[(t.tool == tool) & t.dataset.isin(libs)][col]
    return float(v.mean()) if len(v) else float("nan")


# ---------------------------------------------------------------- numbers the slides quote
PAIR = [("glymph", "g"), ("cellbender", "c")]
# the versions before, where this run scored them beside the current ones: Glymph 0.3.1 (g1), CellBender 0.3.2 (c2)
# and a second run of the current Glymph (g2), for the spread between two runs of the same code
BOTH = PAIR + [(t, tag) for t, tag in (("glymph_031", "g1"), ("cellbender032", "c2"), ("glymph_rep2", "g2")) if (sp.tool == t).any()]
for key, libs in (("all", list(sp.dataset.unique())), ("conf", CONF), ("conf10x", CONF[:7]), ("confxeno", CONF[7:11]), ("confbd", CONF[11:])):
    for tool, tag in BOTH:
        for col, short in (("cross_species_removed", "rem"), ("own_retained", "kept"), ("contam_rmsle", "err"), ("contam_kendall", "rank"),
                           ("same_species_ratio", "ratio")):
            NUM[f"{key}_{tag}_{short}"] = round(mean(tool, col, libs), 4)
    NUM[f"{key}_n"] = len(libs)
for g, libs in GROUPS.items():
    k = g.split()[0].lower()
    for tool, tag in BOTH:
        NUM[f"grp_{k}_{tag}_rem"] = round(mean(tool, "cross_species_removed", libs), 4)
        NUM[f"grp_{k}_{tag}_err"] = round(mean(tool, "contam_rmsle", libs), 4)
    NUM[f"grp_{k}_n"] = len(libs)
p = pd.concat([sp[sp.tool.isin(["glymph", "cellbender"])].pivot(index="dataset", columns="tool", values="contam_rmsle"),
               kid[kid.tool.isin(["glymph", "cellbender"])].pivot(index="dataset", columns="tool", values="contam_rmsle")])
NUM["err_wins"], NUM["err_total"] = int((p.glymph < p.cellbender).sum()), int(len(p))
pr = pd.concat([sp[sp.tool.isin(["glymph", "cellbender"])].pivot(index="dataset", columns="tool", values="contam_kendall"),
                kid[kid.tool.isin(["glymph", "cellbender"])].pivot(index="dataset", columns="tool", values="contam_kendall")])
NUM["rank_wins"] = int((pr.glymph > pr.cellbender).sum())
for tool, tag in BOTH:
    k = kid[kid.tool == tool]
    NUM[f"kid_{tag}_err"], NUM[f"kid_{tag}_rank"] = round(float(k.contam_rmsle.mean()), 4), round(float(k.contam_kendall.mean()), 3)
    NUM[f"kid_{tag}_rot"] = round(float(k.removed_over_truth.mean()), 3)
    NUM[f"kid_{tag}_marker"], NUM[f"kid_{tag}_keep"] = round(float(k.marker_removed_nonpt.mean()), 3), round(float(k.marker_kept_pt.mean()), 3)
    NUM[f"kid_{tag}_n"] = int(len(k))
    r = res[res.tool == tool]
    NUM[f"time_{tag}_median"], NUM[f"time_{tag}_hours"] = round(float(r.wall_s.median())), round(float(r.wall_s.sum() / 3600), 1)
    NUM[f"time_{tag}_n"], NUM[f"ram_{tag}_median"] = int(len(r)), round(float(r.peak_rss_gb.median()), 1)
w = res[res.tool.isin(["glymph", "cellbender"])].pivot(index="dataset", columns="tool", values="wall_s").dropna()
NUM["speedup_median"] = round(float((w.cellbender / w.glymph).median()), 1)
integ = sp[sp.tool != "raw"].groupby("tool")[["n_above_raw", "n_noninteger"]].sum()
NUM["scar_added_entries"] = int(integ.loc["scar", "n_above_raw"]) if "scar" in integ.index else None

# ---------------------------------------------------------------- 1. wall time per library
names = {"glymph": G, "cellbender": B, "scar": "scAR", "soupx": "SoupX", "decontx_bg": "DecontX"}
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for i, (tool, name) in enumerate(names.items()):
    v = res[res.tool == tool].wall_s.sort_values().to_numpy()
    if not len(v):
        continue
    col = T["warm"] if tool == "glymph" else T["cool"] if tool == "cellbender" else T["muted"]
    ax.plot([i + ((j % 7) - 3) * 0.05 for j in range(len(v))], v, ls="", marker="o", ms=4.6, color=col, alpha=0.85, mec="none")
    med = float(pd.Series(v).median())
    ax.hlines(med, i - 0.32, i + 0.32, color=T["ink"], lw=1.8, zorder=4)
    lab = f"{med:,.0f} s" if med < 600 else f"{med / 60:.0f} min"
    ax.annotate(lab, (i + 0.34, med), xytext=(3, 0), textcoords="offset points", va="center", fontsize=12.5, color=T["ink"])
ax.set_yscale("log")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{n}\n{'one A30' if t in ('glymph', 'cellbender', 'scar') else '4 CPU cores'}" for t, n in names.items()], fontsize=12)
ax.set_ylabel("Wall time per library, s (log scale)")
ax.set_xlim(-0.6, len(names) - 0.2)
ax.grid(axis="x", visible=False)
save(fig, "speed_per_library")

# ---------------------------------------------------------------- 2. scaling with library size
sc_path = os.path.join(RUN, "scaling.tsv")
sc = pd.read_csv(sc_path if os.path.exists(sc_path) else os.path.join(RUN0, "scaling.tsv"), sep="\t")
gl = sc[(sc.tool == f"Glymph {GV}") & (sc.state == "completed")]
cb = sc[sc.tool == f"CellBender {CV}"]


def dur(s):
    return f"{int(s // 3600)} h {int(round(s % 3600 / 60))} min" if s >= 3600 else f"{s / 60:.0f} min"


fig, ax = plt.subplots(figsize=(7.6, 4.6))
for t, col, name in ((cb, T["cool"], B), (gl, T["warm"], G)):
    ax.plot(t.cells / 1000, t.wall_s / 60, color=col, lw=2.6, marker="o", ms=8, mec=T["surface"], mew=1.4)
    for c, s_ in zip(t.cells, t.wall_s):
        ax.annotate(dur(s_), (c / 1000, s_ / 60), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=11.5, color=T["ink2"])
    ax.annotate(name, (t.cells.iloc[-1] / 1000, t.wall_s.iloc[-1] / 60), xytext=(12, 0), textcoords="offset points", va="center",
                fontsize=14, color=T["ink"])
ax.set_yscale("log")
ax.set_xlim(8, 196)
ax.set_ylim(1.5, 700)
ax.set_xticks([20, 40, 80, 160])
ax.set_xlabel("Cells in the library, thousands")
ax.set_ylabel("Wall time, min (log scale)")
save(fig, "scaling")
if len(gl):
    NUM["scale_g_160k_s"], NUM["scale_g_160k_ram"] = float(gl.wall_s.iloc[-1]), float(gl.peak_rss_gb.iloc[-1])
    NUM["scale_g_cells"] = [int(c) for c in gl.cells]
NUM["scale_c_160k_s"], NUM["scale_c_160k_ram"] = float(cb.wall_s.iloc[-1]), float(cb.peak_rss_gb.iloc[-1])

# ---------------------------------------------------------------- 3. per-cell error, library by library
kind = {}
for g, libs in GROUPS.items():
    kind.update({d: g for d in libs})
kind.update({d: "Kidney" for d in kid.dataset.unique()})
marks = {"10x cell lines": "o", "BD Rhapsody": "s", "Parse": "D", "Xenograft": "^", "Nuclei mix": "P", "Kidney": "X"}
fig, ax = plt.subplots(figsize=(7.9, 5.2))
hi = float(max(p.glymph.max(), p.cellbender.max())) * 1.08
ax.plot([0, hi], [0, hi], color=T["muted"], lw=1.2, ls=(0, (4, 3)))
ax.fill_between([0, hi], [0, hi], 0, color=T["warm"], alpha=0.07, lw=0)
for g, m in marks.items():
    q = p[[kind.get(d) == g for d in p.index]]
    ax.plot(q.cellbender, q.glymph, ls="", marker=m, ms=8.5, color=T["warm"], mec=T["surface"], mew=1.0, label=f"{g} ({len(q)})")
ax.set_xlim(0, hi)
ax.set_ylim(0, hi)
ax.set_xlabel("CellBender, per-cell error")
ax.set_ylabel("Glymph, per-cell error")
ax.text(hi * 0.97, hi * 0.06, "Glymph more accurate", ha="right", fontsize=12.5, color=T["ink2"])
ax.text(hi * 0.05, hi * 0.93, "CellBender more accurate", ha="left", fontsize=12.5, color=T["ink2"])
ax.legend(loc="center left", fontsize=12, labelcolor=T["ink2"], handletextpad=0.3, bbox_to_anchor=(1.03, 0.5))
ax.set_aspect("equal")
save(fig, "percell_error")

# ---------------------------------------------------------------- 4. removal against counts kept, by platform
panels = [("10x cell lines, never seen", CONF[:7]), ("BD Rhapsody", GROUPS["BD Rhapsody"]), ("Xenograft skin", GROUPS["Xenograft"]),
          ("Nuclei mixture", GROUPS["Nuclei mix"])]
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9))
for ax, (title, libs) in zip(axes, panels):
    for name, tools, col in ((B, ["cellbender_fpr0", "cellbender_fpr05", "cellbender"], T["cool"]),
                             (G, ["glymph_fpr0", "glymph_fpr02", "glymph", "glymph_fpr1"], T["warm"])):
        x = [100 * mean(t_, "own_retained", libs) for t_ in tools]
        y = [100 * mean(t_, "cross_species_removed", libs) for t_ in tools]
        ax.plot(x, y, color=col, lw=2.4, marker="o", ms=7, mec=T["surface"], mew=1.2)
        ax.plot(x[tools.index(name.lower())], y[tools.index(name.lower())], ls="", marker="o", ms=12, mfc="none", mec=col, mew=1.6)
    ax.set_title(f"{title} ({len(libs)})", loc="left", fontsize=13)
    ax.margins(x=0.16, y=0.16)
    ax.set_xlabel("Own counts kept, %")
axes[0].set_ylabel("Cross-species counts removed, %")
fig.tight_layout(w_pad=1.6)
save(fig, "tradeoff_by_platform")

# ---------------------------------------------------------------- 5. cells that lose far more than their background
st_path = os.path.join(RUN, "stripped_cells.tsv")
if os.path.exists(st_path):
    st = pd.read_csv(st_path, sep="\t")
    tot = st.groupby("tool")[["cells", "cells_stripped_far_beyond_truth"]].sum()
    for tool, tag in BOTH + [("scar", "scar"), ("soupx", "soupx"), ("decontx_bg", "decontx")]:
        if tool in tot.index:
            NUM[f"strip_{tag}"] = int(tot.loc[tool, "cells_stripped_far_beyond_truth"])
    NUM["strip_cells"], NUM["strip_libs"] = int(tot.loc["glymph", "cells"]), int(st.dataset.nunique())
    xe = st[st.dataset.str.startswith("xeno")].groupby("tool").cells_stripped_far_beyond_truth.sum()
    for tool, tag in BOTH:
        if tool in xe.index:
            NUM[f"strip_xeno_{tag}"] = int(xe[tool])
    show = [("scar", "scAR", T["muted"]), ("cellbender", f"{B} {CV}", T["cool"]), ("decontx_bg", "DecontX", T["muted"]),
            ("glymph_031", f"{G} 0.3.1", T["muted"]), ("glymph", f"{G} {GV}", T["warm"]), ("soupx", "SoupX", T["muted"])]
    show = [x for x in show if x[0] in tot.index]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (tool, name, col) in enumerate(reversed(show)):
        v = max(float(tot.loc[tool, "cells_stripped_far_beyond_truth"]), 0.8)
        ax.hlines(i, 0.8, v, color=T["hair"], lw=2.2, zorder=1)
        ax.plot([v], [i], ls="", marker="o", ms=11, color=col, mec=T["surface"], mew=1.2, zorder=3)
        ax.annotate(f"{int(tot.loc[tool, 'cells_stripped_far_beyond_truth']):,}", (v, i), xytext=(11, 0), textcoords="offset points", va="center", fontsize=13.5, color=T["ink"])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60000)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([n for _, n, _ in reversed(show)], fontsize=13)
    ax.set_ylim(-0.6, len(show) - 0.4)
    ax.set_xlabel(f"Cells stripped far beyond their background, of {NUM['strip_cells']:,} (log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, "stripped_cells")

json.dump(NUM, open(os.path.join(ROOT, "reports", "deck_figures", "numbers.json"), "w"), indent=1)
print(json.dumps(NUM)[:1500])
