#!/usr/bin/env python
"""Print the depth histogram of a raw library in 0.1 dex bins: barcodes and UMI mass per bin."""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from glymph.io import read_counts  # noqa: E402

for path in sys.argv[1:]:
    N = np.asarray(sp.csr_matrix(read_counts(path).X).sum(1)).ravel()
    edges = np.arange(np.log10(5), np.log10(N.max()) + 0.1, 0.1)
    cnt, _ = np.histogram(np.log10(N[N >= 5]), bins=edges)
    mass = cnt * 10 ** (0.5 * (edges[:-1] + edges[1:]))
    print(os.path.basename(path)[:28])
    print("  N     :", " ".join(f"{10 ** e:6.0f}" for e in edges[:-1]))
    print("  count :", " ".join(f"{c:6d}" for c in cnt))
    print("  mass/k:", " ".join(f"{m / 1e3:6.0f}" for m in mass))
