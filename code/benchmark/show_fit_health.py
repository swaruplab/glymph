#!/usr/bin/env python3
"""One line per library: is the fitted background plausible? (light; no dependencies)

    python3 show_fit_health.py out/TAG [out/OTHER_TAG]

Background at the median depth as a share of that depth, the share of called cells set aside as doublet-like,
whether an amount sits at its bound, and the same for a second run to compare with.
"""
import glob
import json
import os
import sys


def find(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                out.append(v)
            find(v, key, out)
    elif isinstance(o, list):
        for v in o:
            find(v, key, out)
    return out


def line(p):
    r = json.load(open(p))
    g = lambda k: (find(r, k, []) or [None])[0]  # noqa: E731
    bg, ref, kap, kd = g("background_at_median_depth_by_seed"), g("depth_ref"), g("kappa"), g("kappa_depth")
    ref = ref[0] if isinstance(ref, list) else ref
    kap = kap[0] if isinstance(kap, list) else kap
    kd = kd[0] if isinstance(kd, list) else kd
    nc = g("n_cells")
    called = nc[0] if isinstance(nc, list) else nc
    dl = g("doublet_like_cells")
    mid = sorted(bg)[(len(bg) - 1) // 2] if bg else None
    share = None if (mid is None or not ref) else mid / ref
    flag = None if (dl is None or not called) else dl / called
    f = lambda v, n=3: f"{v:.{n}g}" if isinstance(v, (int, float)) else "none"  # noqa: E731
    note = []
    if kap is not None and kap <= 0.1001:
        note.append("kappa at its lower bound")
    if flag is not None and flag > 0.1:
        note.append("over a tenth of the cells set aside")
    if g("reason"):
        note.append(str(g("reason")))
    if g("single_cluster_estimate"):
        note.append("single-cluster estimate")
    return share, flag, f"background {f(share)} of depth, seeds {bg}, kappa {f(kap)}, kappa_d {f(kd)}, ridge {f(g('ridge'))}, set aside {f(flag)}" + (" | " + "; ".join(note) if note else "")


dirs = sys.argv[1:]
for p in sorted(glob.glob(os.path.join(dirs[0], "*.report.json"))):
    L = os.path.basename(p)[: -len(".report.json")]
    out = f"{L:22s} " + line(p)[2]
    for d in dirs[1:]:
        q = os.path.join(d, L + ".report.json")
        if os.path.exists(q):
            s2 = line(q)
            out += f"   || {os.path.basename(d)}: background {'none' if s2[0] is None else format(s2[0], '.3g')}, set aside {'none' if s2[1] is None else format(s2[1], '.3g')}"
    print(out[:330])
