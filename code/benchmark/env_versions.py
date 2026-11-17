#!/usr/bin/env python
"""Print the versions of the named packages in the running environment, then run a check statement.

    python env_versions.py scar torch anndata --check "from scar import model"
"""

import argparse
import importlib.metadata as md
import sys

ap = argparse.ArgumentParser()
ap.add_argument("packages", nargs="*")
ap.add_argument("--check", help="Python statement that must run without error")
a = ap.parse_args()
print("python", sys.version.split()[0])
for p in a.packages:
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "not installed")
try:
    import torch

    print("cuda_build", torch.version.cuda)
except ImportError:
    pass
if a.check:
    exec(a.check)  # noqa: S102  (fails the build if the entry points are missing)
    print("check ok:", a.check[:60])
