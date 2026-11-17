"""End-to-end check of the benchmark path on a simulated two-species library:
simulate -> h5ad -> `glymph run` (CLI) -> species_mix_metrics.py -> metrics table.

    python tests/test_species_mix_e2e.py
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from glymph.cli import main as glymph_main  # noqa: E402
from glymph.simulate import simulate  # noqa: E402


def test_species_mix_end_to_end(extra=()):
    tmp = tempfile.mkdtemp(prefix="glymph_e2e_")
    raw, _ = simulate(species_mix=True, seed=1)
    raw_path, out_path, tsv = (os.path.join(tmp, n) for n in ("raw.h5ad", "glymph.h5ad", "metrics.tsv"))
    raw.write_h5ad(raw_path)
    glymph_main(["run", "--input", raw_path, "--output", out_path, "--device", "cpu",
                 "--target-steps", "800", "--min-epochs", "10", *extra])
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "code", "benchmark", "species_mix_metrics.py"), "--name", "sim",
         "--raw", raw_path, "--tool", f"glymph={out_path}", "--out", tsv, "--per-cell", os.path.join(tmp, "pc")],
        check=True, env=env,
    )
    m = pd.read_csv(tsv, sep="\t").set_index("tool")
    g = m.loc["glymph"]
    assert g.n_above_raw == 0 and g.n_negative == 0 and g.n_noninteger == 0, g
    assert g.cross_species_removed > 0.5, g
    assert g.native_loss < 0.02, g
    assert g.n_missing <= 0.02 * g.n_eval_cells, g
    return m


if __name__ == "__main__":
    print(test_species_mix_end_to_end().T.to_string())
    # the optional paths: exclusion constraint, matched posterior level, gene + cell output rule
    opt = test_species_mix_end_to_end(("--mask-silent", "--match-posterior-mean", "--keep-cell-totals", "--eps-alpha-pinned", "50"))
    print("with --mask-silent --match-posterior-mean --keep-cell-totals:")
    print(opt.loc["glymph", ["cross_species_removed", "own_retained", "native_loss", "contam_rmsle", "contam_kendall"]].to_string())
