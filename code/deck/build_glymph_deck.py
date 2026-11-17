#!/usr/bin/env python3
"""Glymph overview deck: the gap in the field, the idea, the architecture, the benchmark.

    python3 code/deck/make_deck_figures.py dark      # figures and reports/deck_figures/numbers.json
    python3 code/deck/build_glymph_deck.py dark      # reports/Glymph_overview_dark.pptx (light: Glymph_overview.pptx)

Every number on a slide is read from numbers.json or from the run tables, never typed here, except published values
quoted in the gap chapter (sources in the notes of those slides).
"""
import csv
import json

from deck_kit import *  # noqa: F401,F403

N = json.load(open(os.path.join(PRES, "deck_figures", "numbers.json")))
RUN = os.path.join(ROOT, "runs", "20261117-0607_glymph-031-alltier")


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if os.path.exists(path) else []


EXTRA = {r["what"]: r for r in rows(os.path.join(RUN, "resources_extra.tsv"))}


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def dur(s):
    s = float(s)
    if s >= 3600:
        return f"{int(s // 3600)} h {int(round(s % 3600 / 60))} min"
    return f"{s / 60:.0f} min" if s >= 120 else f"{s:.0f} s"


def bal(text, width, size, font=BODY, bold=False):
    """Break a label into lines of even length (forced breaks), so no word is left alone on a last line and the wrap does
    not depend on the renderer. Lines are chosen to minimise the longest one, for the number of lines a greedy wrap needs."""
    if isinstance(text, list):
        return [bal(t, width, size, font, bold) for t in text]
    f = fit_check.font(font, size, bold, False)
    words = text.split(" ")
    wd = [sum(fit_check.width_pt(c, f, 0) for c in w_) for w_ in words]
    sp = fit_check.width_pt(" ", f, 0)
    avail = width * 72 * 0.94
    n, x = 1, 0.0
    for w_ in wd:
        if x > 0 and x + sp + w_ > avail:
            n, x = n + 1, w_
        else:
            x += (sp if x > 0 else 0) + w_
    if n == 1 or len(words) <= n:
        return text
    m = len(words)
    span = lambda i, j: sum(wd[i:j]) + sp * (j - i - 1)  # noqa: E731
    INF = float("inf")
    best = [[INF] * (m + 1) for _ in range(n + 1)]
    cut = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for k in range(1, n + 1):
        for j in range(k, m + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] < INF and span(i, j) <= avail:
                    v = max(best[k - 1][i], span(i, j))
                    if v < best[k][j]:
                        best[k][j], cut[k][j] = v, i
    if best[n][m] == INF:
        return text
    lines, j = [], m
    for k in range(n, 0, -1):
        i = cut[k][j]
        lines.append(" ".join(words[i:j]))
        j = i
    return "\v".join(reversed(lines))


def drow(S, x, y, w, head, text, **kw):
    return dot_row(S.sh, x, y, w, bal(head, w - 0.56, 17, HEAD), bal(text, w - 0.56, 14), **kw)


def card(S, x, y, w, h, icon_name, head, body, head_size=19, size=15, fill="card"):
    body = bal(body, w - 0.56, size)
    g = S.sh.add_group_shape()
    box(g.shapes, x, y, w, h, fill=fill, line="hair", shadow=True, radius=0.07)
    icon_circle(g.shapes, x + 0.28, y + 0.28, 0.5, icon_name)
    hh = text_height([head], head_size, font=HEAD, width=w - 0.56)
    tb(g.shapes, x + 0.28, y + 0.98, w - 0.56, hh + 0.05, head, size=head_size, color="ink", font=HEAD)
    tb(g.shapes, x + 0.28, y + 1.08 + hh, w - 0.56, h - 1.2 - hh, body, size=size, color="ink2", line=1.08, after=4)
    return g


def card_row(S, items, y=1.75, h=3.9, gap=0.3, **kw):
    w = (CW - gap * (len(items) - 1)) / len(items)
    for i, (ic, head, body) in enumerate(items):
        S.anim(card(S, LM + i * (w + gap), y, w, h, ic, head, body, **kw), "rise", gap=220)


def chip_row(S, chips, y, h=1.25, gap=0.3, big_size=26, effect="rise"):
    w = (CW - gap * (len(chips) - 1)) / len(chips)
    for i, (big, label) in enumerate(chips):
        S.anim(chip(S.sh, LM + i * (w + gap), y, w, h, big, bal(label, w - 0.4, 12), big_size=big_size), effect, gap=160)


def key_dot(shapes, x, y, color, label, d=0.16):
    box(shapes, x, y + 0.04, d, d, fill=color, kind=MSO_SHAPE.OVAL)
    return tb(shapes, x + d + 0.1, y - 0.03, 3.2, 0.3, label, size=13, color="ink2")


# ================================================================ title
S = Slide(notes="""Glymph is the lab's ambient RNA remover. The talk has four parts: what is missing in the field, the one idea
behind Glymph, how it is built, and how it compares with every current tool on 54 libraries from three platforms.
All benchmark numbers are for Glymph 0.3.1 against CellBender 0.3.2 unless a slide says otherwise.""", transition="fade", footer=False)
S.anim(S.sh.add_picture(icon("Drop", "gold", 0.10), Inches(8.2), Inches(1.2), Inches(4.9), Inches(4.9)), "zoom", at=100, dur=1100)
S.anim(tb(S.sh, LM, 1.55, 9, 0.35, [[("SWARUP LAB  ·  UCI MIND", {"spc": 3})]], size=11, color="goldtxt", bold=True), "fade")
S.anim(tb(S.sh, LM, 1.95, 9, 1.7, "Glymph", size=96, color="ink", font=HEAD, name="!!title"), "rise", gap=220)
S.anim(tb(S.sh, LM + 0.05, 3.75, 8.6, 1.0, "Measuring the background in single-cell RNA-seq instead of inferring it", size=26,
          color="ink2", font=HEAD, line=1.08), "rise", gap=220)
S.anim(tb(S.sh, LM + 0.05, 5.0, 8.6, 0.5, "A fast ambient RNA remover for 10x, BD Rhapsody and Parse", size=17, color="muted"), "rise", gap=200)
S.anim(tb(S.sh, LM + 0.05, 6.75, 8, 0.3, "Project overview  ·  November 2026", size=11, color="faint"), "fade")
S.finish()

# ================================================================ roadmap
S = Slide("Roadmap", "From a gap in the field to a benchmark", notes="""Four chapters. The gap: why the current standard is not
enough. The idea: measure the amount of background from genes a cell type does not express. The architecture: a small
autoencoder with a count mixture, calibrated by that measurement. The benchmark: 54 libraries, six tools, measured truth.""")
items = [("Binoculars", "01  The gap", "Slow, shaped for one platform, and unable to pin down how much background there is."),
         ("Crosshair", "02  The idea", "Silent genes are negative controls that every library already contains."),
         ("Circuitry", "03  The architecture", "A small autoencoder with a three-part count mixture and integer output."),
         ("Scales", "04  The benchmark", "54 libraries on three platforms, against every current tool.")]
card_row(S, items, y=2.2, h=3.2, head_size=16, size=14.5)
S.takehome("One idea carries the method: measure the background, do not infer it.")
S.finish()

# ================================================================ 1. the gap
divider(1, "The gap", ["Ambient RNA is in every library.", "The standard remover is slow and has to guess its amount."], 0, "Binoculars",
        notes="""Chapter one sets up the problem: what ambient RNA is, how much of it there is, and the three things the field's
standard tool does not give us: speed, a measured amount, and support beyond 10x droplets.""")

S = Slide("The gap  ·  the problem", "Every count has three possible sources", act=0, notes="""A count in the matrix can come from
the cell in the partition, from free RNA in the suspension that every partition captures (ambient RNA), or from a molecule
that changed barcode during amplification (swapped reads). Only the first is biology. Published amounts: background is 3 to 35%
of a cell's counts in mouse kidney, and nuclei from frozen tissue carry medians of 17% and 35% against 3% and 11% in the
matched whole cells (Janssen et al. 2023, Genome Biology). In brain nuclei the ambient RNA is mostly neuronal, which is why
glia show neuronal transcripts.""")
srcs = [("Dna", "The cell itself", "The expression we want to keep."),
        ("Drop", "Ambient RNA", "Free RNA from broken cells, captured by every partition."),
        ("Shuffle", "Swapped reads", "Molecules that change barcode during amplification.")]
for i, (ic, head, body) in enumerate(srcs):
    g = S.sh.add_group_shape()
    y = 1.8 + i * 1.38
    box(g.shapes, LM, y, 5.6, 1.16, fill="card", line="hair", shadow=True)
    icon_circle(g.shapes, LM + 0.25, y + 0.3, 0.56, ic)
    tb(g.shapes, LM + 1.05, y + 0.2, 4.4, 0.4, head, size=18, color="ink", font=HEAD)
    tb(g.shapes, LM + 1.05, y + 0.6, 4.4, 0.5, bal(body, 4.4, 13.5), size=13.5, color="ink2")
    S.anim(g, "rise", gap=200)
    S.anim(arrow(S.sh, LM + 5.7, y + 0.58, LM + 6.75, 3.76), "wipe_l", gap=60)
g = S.sh.add_group_shape()
box(g.shapes, LM + 6.85, 2.96, 2.2, 1.6, fill="feature", line="goldline", shadow=True)
tb(g.shapes, LM + 6.85, 3.05, 2.2, 0.75, "1", size=40, color="onfeature", font=NUM, align="c")
tb(g.shapes, LM + 6.95, 3.92, 2.0, 0.55, "one count in the matrix", size=13, color="onfeature2", align="c")
S.anim(g, "zoom", gap=200)
S.anim(chip(S.sh, W - LM - 2.6, 1.8, 2.6, 1.9, "3 to 35%", bal("of a cell's counts are background in mouse kidney", 2.2, 12), big_size=24), "rise", gap=160)
S.anim(chip(S.sh, W - LM - 2.6, 3.94, 2.6, 1.9, "35%", bal("median in nuclei from frozen kidney, against 11% in matched cells", 2.2, 12), big_size=24), "rise", gap=160)
S.takehome("Background is not a rare artifact: in nuclei it can be a third of every profile.")
S.finish()

S = Slide("The gap  ·  speed", "The standard tool trains a model per library", act=0, notes=f"""CellBender is the reference
method (Nature Methods 2023). It trains a new model from scratch on every library, 150 epochs, on one GPU; its authors report
20 to 60 GPU minutes per library. In our benchmark on one A30 its median was {dur(N['time_c_median'])} per library over
{N['time_c_n']} libraries ({N['time_c_hours']} GPU-hours in total), and a library of 160,000 cells took
{dur(N['scale_c_160k_s'])} with {N['scale_c_160k_ram']:.0f} GB of memory. More GPUs do not shorten a per-library training loop.""")
chip_row(S, [(dur(N["time_c_median"]), "median per library on one A30 GPU, 54 libraries"),
             (dur(N["scale_c_160k_s"]), "for one library of 160,000 cells"),
             (f"{N['scale_c_160k_ram']:.0f} GB", "of memory at that size")], y=1.85, h=1.5, big_size=30)
S.anim(drow(S, LM, 3.85, 5.7, "Trained from scratch every time", "150 epochs of stochastic inference for each library.", icon_name="Hourglass"), "rise", gap=200)
S.anim(drow(S, LM + 6.2, 3.85, 5.7, "One GPU at a time", "A multi-GPU node does not make one library faster.", icon_name="Cpu"), "rise", gap=200)
S.anim(drow(S, LM, 4.95, 5.7, "Fragile defaults", "Learning rate and expected cells often need a second run.", icon_name="Sliders"), "rise", gap=200)
S.anim(drow(S, LM + 6.2, 4.95, 5.7, "Cost grows with the atlas", "Hundreds of libraries mean days of GPU time.", icon_name="Stack"), "rise", gap=200)
S.takehome("At atlas scale, cleaning the data can cost more GPU time than analysing it.")
S.finish()

S = Slide("The gap  ·  identifiability", "The amount of background is not identified", act=0, notes=f"""This is the central gap.
A mixture model explains each count as expression plus background. A flexible decoder can raise the expression of a gene
slightly and lower the background, and fit the data just as well. The likelihood is nearly flat along that direction (the
CellSweep authors say the same of their model), so priors decide the amount. In our benchmark CellBender removed
{pct(N['all_c_ratio'], 0)} of the same-species background expected from the cross-species counts over {N['all_n']} mixtures, and
its per-cell estimates agree with the truth with a rank correlation of {N['all_c_rank']:.2f}. Published benchmarks report weak
per-cell estimates for every tool (Nie et al. 2026).""")
g = S.sh.add_group_shape()
tb(g.shapes, LM, 1.8, 6.4, 0.35, "The same counts of one gene in one cell, read two ways", size=14, color="muted")
for i, (lab, frac) in enumerate((("Reading A", 0.06), ("Reading B", 0.22))):
    y = 2.35 + i * 1.25
    tb(g.shapes, LM, y + 0.14, 1.3, 0.4, lab, size=15, color="ink", font=HEAD)
    wtot = 4.9
    box(g.shapes, LM + 1.4, y, wtot * (1 - frac), 0.62, fill="goldpale", radius=0.08)
    box(g.shapes, LM + 1.4 + wtot * (1 - frac) + 0.05, y, wtot * frac - 0.05, 0.62, fill="gold", radius=0.08)
    tb(g.shapes, LM + 1.55, y + 0.14, 3, 0.4, "expression", size=13, color="ink")
    tb(g.shapes, LM + 1.4, y + 0.68, wtot, 0.3, f"background {int(frac * 100)}%", size=12.5, color="muted", align="r")
S.anim(g, "fade", gap=300, dur=800)
g = S.sh.add_group_shape()
box(g.shapes, LM, 4.95, 6.3, 0.82, fill="card", line="hair")
tb(g.shapes, LM + 0.25, 4.95, 5.9, 0.82, "Both fit the data equally well. The model's priors pick one.", size=15, color="ink", anchor="m")
S.anim(g, "rise", gap=220)
x = W - LM - 5.0
S.anim(chip(S.sh, x, 1.8, 5.0, 1.22, pct(N["all_c_ratio"], 0), bal(f"of the expected same-species background removed by CellBender, {N['all_n']} mixtures", 4.6, 12), big_size=26), "rise", gap=160)
S.anim(chip(S.sh, x, 3.18, 5.0, 1.22, pct(N["grp_nuclei_c_rem"], 0), bal("of cross-species counts removed by CellBender in a nuclei mixture", 4.6, 12), big_size=26), "rise", gap=160)
S.anim(chip(S.sh, x, 4.56, 5.0, 1.22, f"{N['all_c_rank']:.2f}", bal("rank agreement of its per-cell amounts with the truth (1 is perfect)", 4.6, 12), big_size=26), "rise", gap=160)
S.takehome("When the likelihood cannot decide, a prior does. We wanted a measurement.")
S.finish()

S = Slide("The gap  ·  platforms", "No validated remover beyond 10x droplets", act=0, notes="""CellBender assumes one plateau of
empty droplets and one ambient profile per run. That holds for 10x. BD Rhapsody microwells give no plateau: the empty wells are
truncated and unlabelled. Parse split-pool libraries fill the whole barcode space with background, and the background is
structured by the first-round well and by the sublibrary. When this project started, no ambient remover of any kind had been
validated against ground truth on either platform.""")
card_row(S, [("Drop", "10x Chromium", ["Droplets.", "A plateau of empty droplets and one ambient profile per run. Every tool is built on this."]),
             ("SquaresFour", "BD Rhapsody", ["Microwells.", "No plateau to read. The cell threshold itself has to be found differently."]),
             ("Barcode", "Parse Evercode", ["Split-pool barcoding.", "Background differs by first-round well and by sublibrary."])], y=2.15, h=3.25)
S.takehome("One plateau and one profile is a 10x premise. Two major platforms fall outside it.")
S.finish()

S = Slide("The gap  ·  alternatives", "The alternatives break the counts", act=0, notes=f"""The other tools are faster but give up
something else. scAR, the only other neural method, removes the most and also writes counts that exceed the raw data: in our
benchmark {N['scar_added_entries']:,} entries of the species mixtures came back higher than they went in, and published tests
report 17 to 60% artificial counts (Schroeder et al. 2026). SoupX applies one contamination fraction to a whole library and
DecontX needs clusters; both return fractional counts and remove 55 to 80% of cross-species counts in our mixtures.""")
card_row(S, [("Sparkle", "scAR", ["Removes the most.", "Also adds counts that were never sequenced, and can create cell types."]),
             ("Funnel", "SoupX", ["One fraction for the whole library.", "Cannot follow cell-to-cell differences. Fractional output."]),
             ("TreeStructure", "DecontX", ["Needs clusters first.", "Removes about half in cell-line mixtures. Fractional output."])], y=2.15, h=3.25)
S.takehome("To replace the standard, a tool must be fast, measured, and leave whole counts.")
S.finish()

# ================================================================ 2. the idea
divider(2, "The idea", ["Measure the background.", "Do not infer it."], 1, "Crosshair", notes="""Chapter two is the one idea that
separates Glymph from the methods before it: the amount of background is read from the data, from internal negative controls,
and then held fixed while the model does the rest.""")

S = Slide("The idea  ·  silent genes", "Silent genes are built-in negative controls", act=1, notes="""A gene that a cell type does
not express should read zero in those cells. Whatever it does read is background: it is the same logic as reads of another
species in a mixture, or of another genotype in a pooled sample, but it needs no special experiment. Glymph clusters the cells
on the model's own embedding, finds for every cluster the genes whose counts are no more than background alone explains, and
uses the counts on those genes as the measurement. No labels and no marker lists are involved. A typical library gives several
thousand silent genes per cluster.""")
g = S.sh.add_group_shape()
gx, gy, cs = LM + 1.55, 2.2, 0.47
pattern = ["111000000", "000111000", "110000110", "000000111"]
dots = {(0, 4), (0, 7), (1, 0), (1, 8), (2, 3), (2, 5), (3, 1), (3, 4), (0, 5), (1, 6)}
for r, rowp in enumerate(pattern):
    tb(g.shapes, LM, gy + r * (cs + 0.1) + 0.08, 1.45, 0.3, f"Cell type {'ABCD'[r]}", size=13.5, color="ink2")
    for c, v in enumerate(rowp):
        x_, y_ = gx + c * (cs + 0.1), gy + r * (cs + 0.1)
        box(g.shapes, x_, y_, cs, cs, fill="gold" if v == "1" else "surface", line=None if v == "1" else "hair", radius=0.16)
        if (r, c) in dots:
            box(g.shapes, x_ + cs / 2 - 0.045, y_ + cs / 2 - 0.045, 0.09, 0.09, fill="ink2", kind=MSO_SHAPE.OVAL)
tb(g.shapes, gx, gy - 0.42, 9 * (cs + 0.1), 0.3, "genes", size=12.5, color="muted", align="c")
S.anim(g, "fade", gap=300, dur=900)
g = S.sh.add_group_shape()
key_y = gy + 4 * (cs + 0.1) + 0.2
box(g.shapes, gx, key_y, 0.26, 0.26, fill="gold", radius=0.16)
tb(g.shapes, gx + 0.38, key_y - 0.02, 2.2, 0.3, "expressed", size=13, color="ink2")
box(g.shapes, gx + 2.0, key_y, 0.26, 0.26, fill="surface", line="hair", radius=0.16)
box(g.shapes, gx + 2.0 + 0.085, key_y + 0.085, 0.09, 0.09, fill="ink2", kind=MSO_SHAPE.OVAL)
tb(g.shapes, gx + 2.38, key_y - 0.02, 3.4, 0.3, "silent: any count is background", size=13, color="ink2")
S.anim(g, "fade", gap=200)
x = LM + 7.3
S.anim(drow(S, x, 2.0, 4.7, "It should read zero", "A gene a cell type does not express.", num=1), "rise", gap=200)
S.anim(drow(S, x, 3.1, 4.7, "So every count is background", "The same logic as another species in a mixture.", num=2), "rise", gap=200)
S.anim(drow(S, x, 4.2, 4.7, "No labels needed", "Clusters come from the model's own embedding.", num=3), "rise", gap=200)
S.takehome("Every library carries thousands of negative controls. Glymph reads the background off them.")
S.finish()

S = Slide("The idea  ·  what is measured", "Three amounts, each read off the data", act=1, notes="""The measurement is a fit over
single cells of the counts on their silent genes. It separates three amounts. First, ambient RNA that every partition carries,
in units of the level seen in an empty partition. Second, ambient RNA that grows with the depth of the cell, which nuclei carry.
Third, swapped reads, proportional to depth with the average profile of the pool. In the kidney strain-mixture data, where
genotypes give the true background of every nucleus, the truth is a constant of 0.55 empty-partition levels plus 0.47 per unit
of depth; Glymph measured 0.56 to 0.59 plus 0.47 to 0.54 from silent genes alone. Whole cells came out flat, as the truth is.""")
card_row(S, [("Drop", "Ambient, per partition", "A constant amount, in units of what an empty partition holds."),
             ("ChartLineUp", "Ambient, with depth", "Nuclei carry more of it the deeper they are sequenced."),
             ("Shuffle", "Swapped reads", "Proportional to depth, with the average profile of the pool.")], y=1.8, h=2.75)
g = S.sh.add_group_shape()
box(g.shapes, LM, 4.8, CW, 1.1, fill="feature", line="goldline", shadow=True)
tb(g.shapes, LM + 0.35, 4.8, 3.3, 1.1, "Kidney nuclei, checked\vagainst genotypes", size=14, color="onfeature2", anchor="m")
tb(g.shapes, LM + 3.9, 4.8, 3.8, 1.1, [[("truth  ", {"size": 13, "color": "onfeature2"}), ("0.55 + 0.47 × depth", {"font": NUM, "size": 19, "color": "onfeature"})]], anchor="m")
tb(g.shapes, LM + 7.75, 4.8, 4.2, 1.1, [[("measured  ", {"size": 13, "color": "onfeature2"}), ("0.56 + 0.47 × depth", {"font": NUM, "size": 19, "color": "onfeature"})]], anchor="m")
S.anim(g, "rise", gap=260)
S.takehome("The measured amounts match the genotype truth without ever seeing it.")
S.finish()

S = Slide("The idea  ·  per cell", "Then every cell gets its own amount", act=1, notes=f"""Background is not one number per
library: in the kidney truth set the mean per cell is 1.4 to 1.6 times the median. So after the library-level fit, each cell's
own counts on its cluster's silent genes say whether it carries more or less than the trend, and it gets its own multiplier,
shrunk toward the trend when its evidence is thin. The same silent genes also constrain the model: it may not explain counts
on them as expression. On the twelve libraries never used in development, the rank agreement between estimated and true
per-cell background is {N['conf_g_rank']:.2f} for Glymph and {N['conf_c_rank']:.2f} for CellBender.""")
card_row(S, [("Gauge", "A multiplier per cell", "Its own silent-gene counts, compared with what the library trend predicts."),
             ("LockKey", "An exclusion constraint", "The model may not call counts on a cluster's silent genes expression."),
             ("Eye", "Doublet-like cells flagged", "A cell that clearly expresses its silent genes is marked in the output.")], y=1.8, h=2.75)
chip_row(S, [(f"{N['conf_g_rank']:.2f}", "rank agreement with the true per-cell background, Glymph, 12 libraries never seen"),
             (f"{N['conf_c_rank']:.2f}", "the same for CellBender")], y=4.8, h=1.12, big_size=24)
S.takehome("Per-cell amounts are where measuring pays off most.")
S.finish()

# ================================================================ 3. the architecture
divider(3, "The architecture", ["A small autoencoder,", "calibrated by measurement."], 2, "Circuitry", notes="""Chapter three is how
Glymph is built: the pipeline, the model, the output rule, and how one model covers three platforms.""")

S = Slide("The architecture  ·  pipeline", "The pipeline in six steps", act=2, notes="""One: read the whole matrix, cells and empty
partitions. Two: set priors from the depth distribution: the empty level, the cell threshold, the ambient profile, with a
plateau mode for droplets and a knee mode for platforms without a plateau. Three: train the model. Four: cluster the embedding,
find silent genes, and measure the three amounts and the per-cell multipliers. Five: train again briefly with those amounts
fixed and the exclusion constraint on. Six: compute each entry's probability of being background and remove whole counts
gene by gene. Everything runs on one GPU with the library held there as a sparse matrix.""")
steps = [("FolderOpen", "Read", "Every barcode, cells and empties."), ("ChartBar", "Priors", "Empty level and cell threshold from depth."),
         ("Cpu", "Train", "The autoencoder and its mixture."), ("Crosshair", "Measure", "Background on silent genes."),
         ("ArrowsClockwise", "Refit", "With the amounts held fixed."), ("Funnel", "Thin", "Whole counts removed, gene by gene.")]
w, gap = 1.72, 0.342
for i, (ic, head, body) in enumerate(steps):
    x = LM + i * (w + gap)
    g = S.sh.add_group_shape()
    box(g.shapes, x, 2.1, w, 3.1, fill="feature" if i == 3 else "card", line="goldline" if i == 3 else "hair", shadow=True)
    tb(g.shapes, x + 0.2, 2.25, 1.0, 0.4, f"{i + 1:02d}", size=13, color="goldtxt", font=NUM)
    icon_circle(g.shapes, x + (w - 0.62) / 2, 2.75, 0.62, ic)
    tb(g.shapes, x + 0.1, 3.6, w - 0.2, 0.4, head, size=18, color="onfeature" if i == 3 else "ink", font=HEAD, align="c")
    tb(g.shapes, x + 0.16, 4.08, w - 0.32, 1.0, bal(body, w - 0.32, 12.5), size=12.5, color="onfeature2" if i == 3 else "ink2", align="c", line=1.05)
    S.anim(g, "rise", gap=170)
    if i < 5:
        S.anim(arrow(S.sh, x + w + 0.05, 3.65, x + w + gap - 0.05, 3.65), "wipe_l", gap=50)
S.takehome("One GPU, one to four minutes for a typical library. Step four is the new part.")
S.finish()

S = Slide("The architecture  ·  model", "An autoencoder with a three-part count mixture", act=2, notes="""The model is a variational
autoencoder built from plain fully connected layers. The encoder maps a barcode's counts to a low-dimensional embedding; the
decoder turns the embedding into the cell's own expression profile. The expected counts are that profile plus two fixed
profiles: the ambient profile of the barcode's unit, scaled by the measured amount, and the average cell profile of its pool,
scaled by the swapped fraction. The likelihood is a multinomial over the genes that have counts. A literature review at the
start of the project found no evidence that a gene-token transformer helps here; the difficulty is the amount of background,
not the capacity of the network.""")
g = S.sh.add_group_shape()
box(g.shapes, LM, 2.55, 1.9, 1.0, fill="card", line="hair", shadow=True)
tb(g.shapes, LM + 0.15, 2.55, 1.6, 1.0, "counts of\vone barcode", size=13.5, color="ink", align="c", anchor="m")
S.anim(g, "rise", gap=180)
S.anim(arrow(S.sh, LM + 1.95, 3.05, LM + 2.45, 3.05), "wipe_l", gap=50)
for k, (lab, x_) in enumerate((("encoder", LM + 2.5), ("decoder", LM + 6.0))):
    g = S.sh.add_group_shape()
    box(g.shapes, x_, 2.4, 1.7, 1.3, fill="feature", line="goldline", shadow=True)
    tb(g.shapes, x_, 2.4, 1.7, 1.3, lab, size=17, color="onfeature", font=HEAD, align="c", anchor="m")
    S.anim(g, "rise", gap=180)
    if k == 0:
        S.anim(arrow(S.sh, x_ + 1.75, 3.05, x_ + 2.2, 3.05), "wipe_l", gap=50)
        g2 = S.sh.add_group_shape()
        box(g2.shapes, x_ + 2.25, 2.72, 0.95, 0.66, fill="goldtint", line="goldline", radius=0.5)
        tb(g2.shapes, x_ + 2.25, 2.72, 0.95, 0.66, "embedding", size=11.5, color="ink", align="c", anchor="m")
        S.anim(g2, "zoom", gap=120)
        S.anim(arrow(S.sh, x_ + 3.25, 3.05, x_ + 3.45, 3.05), "wipe_l", gap=50)
S.anim(arrow(S.sh, LM + 7.75, 3.05, LM + 8.3, 3.05), "wipe_l", gap=50)
parts = [("The cell's own profile", "from the decoder", 2.05), ("Ambient profile of its unit", "× the measured amount", 3.25), ("Average profile of its pool", "× the swapped fraction", 4.45)]
for lab, sub, y in parts:
    g = S.sh.add_group_shape()
    box(g.shapes, LM + 8.35, y, 3.68, 1.0, fill="card", line="hair", shadow=True)
    tb(g.shapes, LM + 8.55, y + 0.14, 3.3, 0.4, lab, size=15, color="ink", font=HEAD)
    tb(g.shapes, LM + 8.55, y + 0.55, 3.3, 0.35, sub, size=12.5, color="muted")
    S.anim(g, "rise", gap=160)
S.anim(tb(S.sh, LM + 8.35, 5.55, 3.68, 0.35, "summed, then a multinomial over genes", size=12.5, color="goldtxt", italic=True, align="c"), "fade", gap=160)
S.anim(drow(S, LM, 4.4, 7.6, "Small on purpose", "Fully connected layers. The hard part is the amount, and that is measured.", icon_name="Lightbulb"), "rise", gap=200)
S.anim(drow(S, LM, 5.25, 7.6, "The library lives on the GPU", "A sparse matrix on the device: no data loader in the loop.", icon_name="Lightning"), "rise", gap=200)
S.takehome("A small model is enough once the amount of background is no longer its job.")
S.finish()

S = Slide("The architecture  ·  output", "Whole numbers, never above the raw count", act=2, notes="""The output rule matters as much
as the model. For every gene, the number of counts to remove is the expected background of that gene summed over cells; they
are taken from the entries where background is most probable. A false-positive budget, 0.4% by default, lets a gene give up a
little more than expected and is the one dial that trades removal against counts kept. For libraries up to 150 million entries
the rule also keeps each cell's expected background total. The result is an integer matrix in which no entry exceeds the raw
count, so every downstream tool sees ordinary counts.""")
card_row(S, [("Target", "A budget per gene", "Remove each gene's expected background from the entries where it is most probable."),
             ("Sliders", "One dial", "A false-positive budget, 0.4% by default, trades removal against counts kept."),
             ("ShieldCheck", "Only subtraction", "Integers in, integers out. No entry is ever higher than it was.")], y=1.8, h=2.75)
chip_row(S, [("0", "entries above the raw count, all 54 libraries"), ("0", "fractional entries"),
             (f"{N['scar_added_entries'] / 1e6:.0f} million", "entries raised by scAR on the species mixtures")], y=4.8, h=1.12, big_size=24)
S.takehome("Glymph only subtracts. Downstream tools see ordinary count matrices.")
S.finish()

S = Slide("The architecture  ·  platforms", "One model, three platforms", act=2, notes="""Platform structure enters as two labels per
barcode. The unit says which barcodes share ambient RNA: the library for 10x and BD Rhapsody, the first-round well for Parse.
The pool says which barcodes were amplified together: the sublibrary for Parse. Where there is no plateau of empty partitions,
the priors come from the knee of the depth distribution, and a test rejects a false plateau made of cells. Without any vendor
cell calls, Glymph's cell count on a BD Rhapsody spleen library is 8,061; the vendor pipeline calls 8,061.""")
card_row(S, [("Drop", "10x Chromium", ["One unit per library.", "Priors from the plateau of empty droplets."]),
             ("SquaresFour", "BD Rhapsody", ["One unit per cartridge.", "Priors from the knee: there is no plateau to read."]),
             ("Barcode", "Parse Evercode", ["A unit per first-round well, a pool per sublibrary.", "Each well has its own ambient profile."])], y=1.8, h=2.95)
chip_row(S, [("8,061 = 8,061", "cells called by Glymph and by the vendor on a BD Rhapsody spleen library, with no vendor input")], y=4.95, h=1.0, big_size=22)
S.takehome("Two labels per barcode: who shares ambient RNA, and who was amplified together.")
S.finish()

# ================================================================ 4. the benchmark
divider(4, "The benchmark", ["54 libraries, three platforms,", "six tools, measured truth."], 3, "Scales", notes="""Chapter four is the
evidence. Every tool ran on every library on the same GPU model, and every score comes from a truth that does not depend on
any tool.""")

S = Slide("The benchmark  ·  design", "54 libraries with four kinds of truth", act=3, notes="""The libraries: 27 human and mouse
mixtures (cell lines on 10x, BD Rhapsody and Parse, a nuclei mixture, and six xenografts), where every count of the other
species is background; five mouse kidney libraries pooled from three strains, where genotypes give the background of every
cell (Janssen et al. 2023); eight Parse libraries with one tissue per well; and 14 libraries without truth, scored on count
integrity and marker leakage. Tools: Glymph, CellBender 0.3.2, scAR, SoupX, and DecontX with and without an empty-droplet
background. Twelve mixtures were never used to choose anything in Glymph 0.3.1. Genes that collect reads aligned to the wrong
genome, such as ribosomal RNA on mouse Gm42418, are masked for every tool by a rule that uses raw counts only.""")
chip_row(S, [("54", "libraries"), ("3", "platforms: 10x, BD Rhapsody, Parse"), ("6", "tools on the same GPU model"), ("12", "mixtures never seen in development")],
         y=1.8, h=1.2, big_size=30)
card_row(S, [("Atom", "Species mixtures", "27 libraries. Every count from the other species is background."),
             ("Fingerprint", "Genotypes", "5 kidney libraries. Strain alleles give each cell's true background."),
             ("MapTrifold", "Tissue by well", "8 Parse libraries with one tissue per first-round well."),
             ("Question", "No truth", "14 libraries, scored on integrity and marker leakage.")], y=3.2, h=2.75, head_size=17, size=13.5)
S.takehome("Twelve libraries were never used to choose anything. They are the test.")
S.finish()

sp_x = round(N["speedup_median"])
S = Slide("The benchmark  ·  speed", f"About {sp_x} times faster per library", act=3, notes=f"""Wall time of the whole command on one
A30 GPU, one library per run, {N['time_g_n']} libraries. Glymph's median is {dur(N['time_g_median'])} and CellBender's
{dur(N['time_c_median'])}; the median ratio across libraries is {N['speedup_median']}. In total Glymph needed {N['time_g_hours']}
GPU-hours for the benchmark and CellBender {N['time_c_hours']}. Median memory is {N['ram_g_median']} GB against
{N['ram_c_median']} GB. SoupX and DecontX run on four CPU cores and are as fast as Glymph, but they are much less accurate."""
          + (f""" All 54 libraries on one node with four A30s: {dur(EXTRA['All 54 libraries in one job']['wall_s'])}.""" if "All 54 libraries in one job" in EXTRA else ""))
fig_right_chips(S, "speed_per_library.png", [(dur(N["time_g_median"]), "Glymph, median per library"),
                                               (dur(N["time_c_median"]), "CellBender, median per library"),
                                               (f"{N['time_g_hours']} h", f"GPU time for all 54 libraries, against {N['time_c_hours']} h")],
                f"The whole benchmark takes Glymph about {round(N['time_g_hours'])} GPU-hours; CellBender needs {round(N['time_c_hours'])}.", fmt=bal)
S.finish()

S = Slide("The benchmark  ·  scale", "160,000 cells in minutes, not hours", act=3, notes=f"""One library of 20,000 cells copied one
to eight times, on one A30 with 24 GB. Used for time and memory only, never accuracy. At 160,000 cells Glymph takes
{dur(N.get('scale_g_160k_s', 0))} and {N.get('scale_g_160k_ram', 0):.0f} GB of memory; CellBender takes {dur(N['scale_c_160k_s'])}
and {N['scale_c_160k_ram']:.0f} GB. Glymph sums over the library in pieces, keeps gene indices in 32 bits, and writes the output
gene block by gene block, so GPU memory stays under 19 GB of the card's 24.""")
fig_right_chips(S, "scaling.png", [(dur(N.get("scale_g_160k_s", 0)), "Glymph at 160,000 cells"),
                                     (dur(N["scale_c_160k_s"]), "CellBender at 160,000 cells"),
                                     (f"{N.get('scale_g_160k_ram', 0):.0f} GB", f"memory, against {N['scale_c_160k_ram']:.0f} GB")],
                "Library size is no longer a reason to split the data or to wait overnight.", fmt=bal)
S.finish()

S = Slide("The benchmark  ·  accuracy", f"More accurate per cell on {N['err_wins']} of {N['err_total']} libraries", act=3, keep_alias=True,
          notes=f"""Each point is one library with truth: the 27 mixtures and the five kidney libraries. The axes are the per-cell
error, the root mean squared log error (RMSLE) between each cell's estimated and true background fraction; lower is better,
and points under the dashed line are libraries where Glymph is more accurate. Glymph has the lower error on {N['err_wins']} of
{N['err_total']} and the higher rank agreement on {N['rank_wins']}. On the twelve libraries never used in development the mean error
is {N['conf_g_err']:.3f} against {N['conf_c_err']:.3f}, and the rank agreement {N['conf_g_rank']:.2f} against {N['conf_c_rank']:.2f}.""")
fig_right_chips(S, "percell_error.png", [(f"{N['err_wins']} of {N['err_total']}", "libraries with truth where Glymph has the lower error"),
                                          (f"{N['conf_g_err']:.3f}", f"mean error on 12 libraries never seen; CellBender {N['conf_c_err']:.3f}"),
                                          (f"{N['conf_g_rank']:.2f}", f"rank agreement on the same 12; CellBender {N['conf_c_rank']:.2f}")],
                "Glymph knows which cells are dirty: about half of CellBender's per-cell error.", fig_w=7.3, chip_w=4.3, fmt=bal)
S.finish()

S = Slide("The benchmark  ·  removal", "Level on cell lines, far ahead on nuclei", act=3, notes=f"""Each panel is one group of
mixtures. Up is more cross-species counts removed; right is more of the cells' own counts kept. Each line follows a tool's
false-positive budget, and the ring marks its default. On the seven 10x cell-line mixtures never used in development the two
tools are level: Glymph removes {pct(N['conf10x_g_rem'])} and CellBender {pct(N['conf10x_c_rem'])}. On BD Rhapsody Glymph removes
{pct(N['grp_bd_g_rem'])} and CellBender {pct(N['grp_bd_c_rem'])}. On the nuclei mixture Glymph removes {pct(N['grp_nuclei_g_rem'])}
and CellBender {pct(N['grp_nuclei_c_rem'])}. Keeping all of a cell's own counts is not the aim: they also contain background of
the same species, and CellBender leaves a quarter of it in.""")
S.anim(pic(S.sh, "tradeoff_by_platform.png", LM, 1.62, CW, 3.75), "fade", gap=300, dur=900)
g = S.sh.add_group_shape()
key_dot(g.shapes, LM + 0.2, 5.6, "warm", "Glymph")
key_dot(g.shapes, LM + 1.6, 5.6, "cool", "CellBender")
tb(g.shapes, LM + 3.5, 5.57, 8.4, 0.3, "Ring: default setting. Along a line the false-positive budget rises.", size=13, color="muted")
S.anim(g, "fade", gap=200)
S.takehome("Same removal as CellBender on 10x cell lines, and more than twice as much on nuclei.")
S.finish()

S = Slide("The benchmark  ·  tissue", "In kidney tissue, closer to the genotype truth", act=3, notes=f"""Five mouse kidney libraries,
three of whole cells and two of nuclei, pooled from three strains (Janssen et al. 2023). The authors' genotype-based estimate
gives the true background fraction of every cell. Glymph was developed on these five libraries, so this slide is a check, not
a test. Per-cell error {N['kid_g_err']:.3f} against {N['kid_c_err']:.3f}; rank agreement {N['kid_g_rank']:.2f} against
{N['kid_c_rank']:.2f}; total removed over the true total {N['kid_g_rot']:.2f} against {N['kid_c_rot']:.2f}. Proximal tubule cells
dominate the ambient RNA here: Glymph removes {pct(N['kid_g_marker'], 0)} of tubule-marker counts from other cell types and keeps
{pct(N['kid_g_keep'], 0)} of them in tubule cells; CellBender {pct(N['kid_c_marker'], 0)} and {pct(N['kid_c_keep'], 0)}.""")
table = [("Per-cell error", f"{N['kid_g_err']:.3f}", f"{N['kid_c_err']:.3f}", "lower", N["kid_g_err"] < N["kid_c_err"]),
         ("Rank agreement across cells", f"{N['kid_g_rank']:.2f}", f"{N['kid_c_rank']:.2f}", "higher", N["kid_g_rank"] > N["kid_c_rank"]),
         ("Removed, as a share of the true background", f"{N['kid_g_rot']:.2f}", f"{N['kid_c_rot']:.2f}", "near 1",
          abs(N["kid_g_rot"] - 1) < abs(N["kid_c_rot"] - 1)),
         ("Tubule markers removed from other cell types", pct(N["kid_g_marker"], 0), pct(N["kid_c_marker"], 0), "higher", N["kid_g_marker"] > N["kid_c_marker"]),
         ("Tubule markers kept in tubule cells", pct(N["kid_g_keep"], 0), pct(N["kid_c_keep"], 0), "higher", N["kid_g_keep"] > N["kid_c_keep"])]
g = S.sh.add_group_shape()
tb(g.shapes, LM + 6.4, 1.75, 1.9, 0.3, "Glymph", size=13, color="muted", align="c", bold=True)
tb(g.shapes, LM + 8.4, 1.75, 1.9, 0.3, "CellBender", size=13, color="muted", align="c", bold=True)
tb(g.shapes, LM + 10.4, 1.75, 1.6, 0.3, "Better is", size=13, color="muted", align="c", bold=True)
S.anim(g, "fade", gap=160)
for i, (lab, a, b, better, g_wins) in enumerate(table):
    y = 2.15 + i * 0.76
    g = S.sh.add_group_shape()
    box(g.shapes, LM, y, CW, 0.64, fill="card", line="hair", radius=0.18)
    tb(g.shapes, LM + 0.3, y, 6.0, 0.64, lab, size=15, color="ink", anchor="m")
    tb(g.shapes, LM + 6.4, y, 1.9, 0.64, a, size=19, color="ink", font=NUM, align="c", anchor="m")
    tb(g.shapes, LM + 8.4, y, 1.9, 0.64, b, size=19, color="ink2", font=NUM, align="c", anchor="m")
    tb(g.shapes, LM + 10.4, y, 1.6, 0.64, better, size=13, color="muted", align="c", anchor="m")
    xm = LM + (6.4 if g_wins else 8.4) + 0.3
    g.shapes.add_picture(icon("CheckCircle", "gold"), Inches(xm), Inches(y + 0.19), Inches(0.26), Inches(0.26))
    S.anim(g, "rise", gap=150)
S.takehome("Where genotypes give the truth for every cell, Glymph is as close or closer.")
S.finish()

S = Slide("The benchmark  ·  limits", "Where Glymph does not win yet", act=3, notes=f"""Three honest limits. First, clean tissue. On
the four xenograft libraries never used in development Glymph removes {pct(N['confxeno_g_rem'], 0)} of cross-species counts against
{pct(N['confxeno_c_rem'], 0)}, but it also removes {N['confxeno_g_ratio']:.1f} times the expected same-species background and keeps
{pct(N['confxeno_g_kept'])} of the cells' own counts against {pct(N['confxeno_c_kept'])}. Those libraries have so little background
that only one to five genes can be tested, and the measured level is too high. Second, BD Rhapsody: on the one mixture never
used in development Glymph removes {pct(N['confbd_g_rem'])} and CellBender {pct(N['confbd_c_rem'])}. Third, repeat runs on such thin
libraries differ by up to eight points of removal. One kidney library is also cleaned 1.2 to 1.3 times beyond the truth, and on
Parse tissue libraries Glymph keeps fewer counts of cluster-specific marker genes in their own cluster than CellBender does
(94% against 99%), a sign of the same over-cleaning.""")
card_row(S, [("Plant", "Clean tissue", [f"Xenografts: {pct(N['confxeno_g_rem'], 0)} removed against {pct(N['confxeno_c_rem'], 0)}.",
                                         f"But {N['confxeno_g_ratio']:.0f} times the expected background of the cells' own species goes with it."]),
             ("SquaresFour", "BD Rhapsody removal", [f"{pct(N['confbd_g_rem'])} against {pct(N['confbd_c_rem'])} on the mixture never seen.",
                                                     "Per-cell error is a quarter of CellBender's."]),
             ("ArrowsClockwise", "Repeat runs", ["Libraries with very little background give few silent genes to test.",
                                                 "Their result moves by up to 8 points between runs."])], y=2.0, h=3.6)
S.takehome("Very clean libraries are the open problem: too few silent genes to measure with.")
S.finish()

S = Slide("Summary", "Glymph in one slide", act=3, notes=f"""Glymph measures the amount of background from silent genes, per library
and per cell, and holds it fixed while a small autoencoder separates expression from background. It is about {sp_x} times
faster than CellBender per library, more accurate per cell on {N['err_wins']} of {N['err_total']} libraries with truth, level on
removal for 10x cell lines, far ahead on nuclei, and it runs on 10x, BD Rhapsody and Parse with integer output. Next: a more
robust level for very clean libraries, a pretrained model that needs no training per library, and more truth data for Parse.""")
chip_row(S, [(f"{sp_x}×", "faster per library than CellBender"), (f"{N['err_wins']} of {N['err_total']}", "libraries with truth where it is more accurate per cell"),
             ("3", "platforms: 10x, BD Rhapsody, Parse"), ("0", "counts added, ever")], y=1.85, h=1.45, big_size=32)
S.anim(drow(S, LM, 3.75, 5.7, "The idea", "Silent genes measure the background; nothing is left to a prior.", icon_name="Crosshair"), "rise", gap=200)
S.anim(drow(S, LM + 6.2, 3.75, 5.7, "The model", "A small autoencoder with a count mixture and integer output.", icon_name="Circuitry"), "rise", gap=200)
S.anim(drow(S, LM, 4.85, 5.7, "The evidence", "54 libraries, six tools, twelve libraries never seen.", icon_name="Scales"), "rise", gap=200)
S.anim(drow(S, LM + 6.2, 4.85, 5.7, "Next", "Very clean libraries, a pretrained model, more Parse truth.", icon_name="Rocket"), "rise", gap=200)
S.takehome("Measure the background. Do not infer it.")
S.finish()

finalize("Glymph_overview", len(SLIDES), "Glymph: measuring the background in single-cell RNA-seq",
         "Project overview: the gap, the idea, the architecture, the benchmark")
