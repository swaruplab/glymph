#!/usr/bin/env python
"""Gather the all-tier benchmark into a few tables under <run>/out/deliverables/.

    python collect_all_tiers.py <run dir> [config/datasets.tsv]

  scores_species.tsv   one row per library and tool, species truth (species_mix_metrics.py),
                       alignment-artifact genes masked; scores_species_unmasked.tsv counts every gene and
                       artifact_genes.tsv lists what was masked in each library
  scores_kidney.tsv    one row per kidney library and tool (kidney_metrics.py)
  scores_generic.tsv   one row per library and tool, truth-free (generic_metrics.py)
  resources.tsv        one row per library and tool: wall time, CPU core time, peak RAM, GPU memory
                       and utilisation, node and GPU model (from profile.sh), with the library size
  summary.tsv          means per tier group and tool, and separately over the held-out libraries;
                       n says how many libraries each mean is over
  failures.tsv         library and tool pairs that have a profile but no scored output
Every table carries tier, platform, truth and how the library was used in development.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import parse_profile  # noqa: E402

run = sys.argv[1]
cfg = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "config", "datasets.tsv"), sep="\t", dtype=str).fillna("")
meta = cfg[["name", "tier", "platform", "kind", "truth", "development"]].rename(columns={"name": "dataset"})
out = os.path.join(run, "out", "deliverables")
os.makedirs(out, exist_ok=True)
SKIP = ("metrics", "generic", "deliverables", "prep", "envs", "diag", "scaling")


def read_all(pattern):
    fs = sorted(glob.glob(os.path.join(run, pattern)))
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True) if fs else pd.DataFrame()


met = read_all("out/metrics/*.tsv")
gen = read_all("out/generic/*.tsv")
species = met[met.dataset.isin(cfg.name[cfg.truth == "species"])].dropna(axis=1, how="all") if len(met) else met
kidney = met[met.dataset.isin(cfg.name[cfg.truth == "genotype"])].dropna(axis=1, how="all") if len(met) else met
unm = read_all("out/metrics_unmasked/*.tsv")
art = read_all("out/metrics/artifacts/*.tsv")
if len(art):
    art.to_csv(os.path.join(out, "artifact_genes.tsv"), sep="\t", index=False, float_format="%.4g")
for name, t in (("scores_species", species), ("scores_species_unmasked", unm), ("scores_kidney", kidney), ("scores_generic", gen)):
    if len(t):
        t.merge(meta, on="dataset", how="left").to_csv(os.path.join(out, name + ".tsv"), sep="\t", index=False, float_format="%.6g")

# resources: every profile under out/<tool>/
rows = []
for f in sorted(glob.glob(os.path.join(run, "out", "*", "*.time.txt"))):
    tool = os.path.basename(os.path.dirname(f))
    if tool.startswith(SKIP):
        continue
    name = os.path.basename(f)[: -len(".time.txt")]
    r = {"dataset": name, "tool": tool}
    r.update(parse_profile(f[: -len(".time.txt")]))
    rows.append(r)
res = pd.DataFrame(rows)
if len(res):
    # SoupX and the two DecontX runs share one R profile per tool name already; keep as recorded.
    size = gen[gen.tool == "raw"][["dataset", "n_eval_cells"]] if len(gen) else pd.DataFrame(columns=["dataset", "n_eval_cells"])
    res = res.merge(meta, on="dataset", how="left").merge(size, on="dataset", how="left")
    res.to_csv(os.path.join(out, "resources.tsv"), sep="\t", index=False, float_format="%.6g")

# failures: a profile without a score
scored = set(zip(gen.dataset, gen.tool)) if len(gen) else set()
fail = [r for r in rows if (r["dataset"], r["tool"]) not in scored]
pd.DataFrame(fail)[["dataset", "tool"] + [c for c in ("wall_s", "node") if fail and c in fail[0]]].to_csv(
    os.path.join(out, "failures.tsv"), sep="\t", index=False) if fail else open(os.path.join(out, "failures.tsv"), "w").write("dataset\ttool\n")

# summary
GROUP = {"t1_tuning": "cell-line mixtures, tuning (10x)", "t1_holdout": "cell-line mixtures, other (10x)", "t1_nuclei": "nuclei mixture (10x)",
         "t2_kidney": "kidney strain mixture (10x)", "t2_xeno": "xenograft skin (10x)", "t3_rhapsody": "BD Rhapsody", "t4_parse": "Parse Evercode",
         "t5_notruth": "cortex nuclei (10x)"}
parts = []


def add(t, cols, kind):
    if not len(t):
        return
    t = t.drop(columns=[c for c in meta.columns if c != "dataset" and c in t]).merge(meta, on="dataset", how="left")
    t = t[t.tool != "raw"]
    for label, sel in (("all", t), ("held out", t[t.development == "held out"])):
        if not len(sel):
            continue
        g = sel.groupby([sel.tier.map(GROUP).fillna(sel.tier), "tool"])
        m = g[[c for c in cols if c in sel]].mean()
        m.insert(0, "n", g.size())
        m = m.reset_index().rename(columns={"tier": "group"})
        m.insert(0, "libraries", label)
        m.insert(0, "scores", kind)
        parts.append(m)


add(species, ["cross_species_removed", "own_retained", "native_loss", "same_species_ratio", "contam_rmsle", "contam_kendall", "n_above_raw", "n_noninteger"], "species")
add(unm, ["cross_species_removed", "own_retained", "native_loss", "same_species_ratio", "contam_rmsle", "contam_kendall"], "species, artifact genes counted")
add(kidney, ["contam_rmsle", "contam_kendall", "removed_over_truth", "marker_removed_nonpt", "marker_kept_pt", "n_above_raw", "n_noninteger"], "kidney")
add(gen, ["removed_pct", "marker_removed_off", "marker_kept_on", "n_missing", "n_above_raw", "n_noninteger"], "generic")
if len(res):
    add(res.drop(columns=[c for c in meta.columns if c != "dataset"]), ["wall_s", "cpu_core_s", "peak_rss_gb", "gpu_peak_mem_gb", "gpu_mean_util_pct"], "resources")
if parts:
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False, float_format="%.5g")
print(f"[collect] species rows {len(species)}, kidney rows {len(kidney)}, generic rows {len(gen)}, profiles {len(res)}, unscored profiles {len(fail)} -> {out}")
