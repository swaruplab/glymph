#!/usr/bin/env python3
"""Print chosen diagnostics from Glymph report files (light; no dependencies).

    python3 show_report_keys.py key1,key2,... report.json [report.json ...]
"""
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


keys = sys.argv[1].split(",")
for path in sys.argv[2:]:
    try:
        r = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        print(f"{path}: {e}")
        continue
    print("## " + "/".join(path.split(os.sep)[-2:]))
    for k in keys:
        v = find(r, k, [])
        if v:
            print(f"  {k}: {v[0] if len(v) == 1 else v}"[:1500])
