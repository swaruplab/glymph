#!/usr/bin/env python3
"""One line per library and run: the three fitted background amounts and how the swapped share was chosen (light; no dependencies).

    python3 show_fit_table.py out/TAG1 out/TAG2 ...
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


libs = sorted({os.path.basename(f)[: -len(".report.json")] for d in sys.argv[1:] for f in glob.glob(os.path.join(d, "*.report.json"))})
for L in libs:
    for d in sys.argv[1:]:
        p = os.path.join(d, L + ".report.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        g = lambda k: (find(r, k, []) or [None])  # noqa: E731
        fmt = lambda v: "none" if v is None else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))  # noqa: E731
        ks, kd = g("kappa"), g("kappa_depth")
        ss = [v for v in g("s") if isinstance(v, float)]
        tab = g("pairs_on_the_ridge_by_share")
        t = "" if tab[0] is None else " ridge pairs " + " ".join(str(x[1]) for x in tab[0])
        print(f"{L:20s} {os.path.basename(d):12s} kappa {fmt(ks[0])} kappa_d {fmt(kd[0])} s {fmt(ss[0] if ss else None)} ridge {fmt(g('ridge')[0])} clusters {fmt(g('n_clusters')[0])} "
              f"swap share {fmt(g('swap_share_of_depth_proportional')[0])}{t}")
