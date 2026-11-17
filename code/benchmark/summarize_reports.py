#!/usr/bin/env python
"""One line per Glymph run report in a folder: thresholds, calibration, cells, share removed, time.

    python summarize_reports.py out/<tool folder> [more folders]
"""
import glob
import json
import os
import sys

for folder in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(folder, "*.report.json"))):
        r = json.load(open(f))
        p, c = r["priors"], r["swap_calibration"]
        pools = c.get("pools", {})
        num = lambda v, k: round(float(v), k) if isinstance(v, (int, float)) else None  # noqa: E731
        ks = [num(v, 3) for v in c.get("kappa_estimate", [])]
        ss = [num(v, 4) for v in c.get("swap_estimate", [])]
        if len(ks) > 3:
            ok = [v for v in ks if v is not None]
            ks = f"{len(ok)}/{len(ks)} pools, median {sorted(ok)[len(ok) // 2] if ok else None}"
            oks = [v for v in ss if v is not None]
            ss = f"median {sorted(oks)[len(oks) // 2] if oks else None}"
        p0 = pools.get("0", {})
        pc = c.get("per_cell", {}).get("0", {})
        kd = p0.get("kappa_depth")
        used = [x for x in p0.get("clusters", []) if x.get("used")]
        t = r["timings"]
        print(f"{os.path.basename(f)[:-12]:22s} {p['threshold_method'][:7]:7s} P {p['empty_level']:6.0f} T {p['cell_threshold']:6.0f} "
              f"conf {p.get('confident_threshold', 0):6.0f} n_conf {c.get('n_confident_cells', 0):6d} cells {r['n_cells']:6d} | "
              f"kappa {ks} depth {None if not isinstance(kd, (int, float)) else round(kd, 3)} s {ss} alpha {num(pc.get('alpha'), 2)} spread {pc.get('multiplier_over_trend_q10_q50_q90')} learned {[round(v, 4) for v in c.get('swap_learned', [])][:3]} | "
              f"clusters {len(used)}/{p0.get('n_clusters')} dropped {p0.get('clusters_dropped')} trimmed {p0.get('cells_trimmed')}/{p0.get('cells_fitted')} "
              f"{p0.get('reason', '')} | removed {100 * r['removed_fraction_in_cells']:.1f}% | "
              f"{sum(v for k, v in t.items() if k.endswith('_s')):.0f} s, {r['peak_gpu_mem_gb']} GB")
