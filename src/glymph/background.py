"""Closed-form background estimates and priors, computed before any training.

Glymph never asks which vendor made a library. It asks two things about each barcode:
  unit: which barcodes share a soup (10x: the library; Parse: round 1 well or sample;
        BD Rhapsody: the cartridge lane)
  pool: which barcodes were amplified together and so share chimeric / swapped reads
        (10x: the library; Parse: the sublibrary; Rhapsody: the lane)

Everything here is a heuristic that only has to be roughly right: the model refines
the cell calls and the ambient amount. The thresholds are written to the report so
they can be inspected, and each can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Priors:
    cell_threshold: float  # UMI count above which a barcode is very likely a cell
    empty_level: float  # typical UMI count of a cell-free partition
    empty_max: float  # upper edge of the "surely background" window
    analyzed: np.ndarray  # row indices (into the full matrix) that the model sees
    genes: np.ndarray  # column indices kept (nonzero in the analyzed barcodes)
    prior_cell_logit: np.ndarray  # per analyzed barcode
    unit: np.ndarray  # unit code per analyzed barcode
    pool: np.ndarray  # pool code per analyzed barcode
    unit_names: list
    pool_names: list
    soup: np.ndarray  # [n_units, n_genes] background profile per unit
    chibar: np.ndarray  # [n_pools, n_genes] average cell profile per pool
    e_unit: np.ndarray  # [n_units] ambient UMIs per partition
    d_loc: float  # log-normal prior on native cell size
    d_scale: float
    diagnostics: dict = field(default_factory=dict)


def ordmag_threshold(n_sorted: np.ndarray, expected_cells: int | None = None, low: float = 5.0):
    """Order-of-magnitude cell threshold with automatic choice of the expected cell number.

    For a candidate cell number x the threshold is one tenth of the 99th percentile of the
    top x barcodes. x is chosen to minimise (n_called(x) - x)^2 / x, the rule Cell Ranger
    uses when the expected number of cells is not given.
    """
    n_valid = int((n_sorted >= low).sum())
    if n_valid == 0:
        raise ValueError("No barcode reaches the low-count threshold.")

    def thr(x):
        return max(float(np.quantile(n_sorted[:x], 0.99)) / 10.0, low)

    def called(t):
        return int(np.searchsorted(-n_sorted, -t, side="right"))

    if expected_cells:
        t = thr(min(int(expected_cells), n_valid))
        return t, called(t)
    hi = min(n_valid, 262144)
    cands = np.unique(np.round(np.geomspace(min(50, hi), hi, 300)).astype(int))
    best = (np.inf, None, None)
    for x in cands:
        t = thr(int(x))
        n = called(t)
        loss = (n - x) ** 2 / x
        if loss < best[0]:
            best = (loss, t, n)
    return best[1], best[2]


def empty_mode(N: np.ndarray, cell_threshold: float, low: float) -> float:
    """UMI-weighted mode of log10(counts) among barcodes well below the cell threshold.

    Weighting by UMIs asks "where does most of the non-cell RNA sit", which is the
    empty-partition plateau when one exists and suppresses the long tail of barcodes
    with a handful of counts.
    """
    top = cell_threshold / 2.0
    sub = N[(N >= low) & (N < top)]
    if sub.size < 100 or top <= low * 1.5:
        return float(np.median(sub)) if sub.size else float(low)
    edges = np.arange(np.log10(low), np.log10(top) + 0.05, 0.05)
    hist, _ = np.histogram(np.log10(sub), bins=edges, weights=sub)
    kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    smooth = np.convolve(hist, kern / kern.sum(), mode="same")
    i = int(np.argmax(smooth))
    return float(10 ** (0.5 * (edges[i] + edges[i + 1])))


def plateau_mixture(N: np.ndarray, low: float = 5.0, bin_width: float = 0.05):
    """Find the empty-partition plateau and the cells from the depth distribution alone.

    Droplet libraries hold three kinds of barcode: cell-containing partitions, cell-free
    partitions that still have a bead and so collect ambient RNA (the plateau), and a long
    tail of barcodes with a handful of counts. The order-of-magnitude rule assumes a cliff
    between cells and plateau. In nuclei from frozen tissue there is none: the plateau sits
    at several hundred counts, right under the nuclei. That rule then puts the cell threshold
    inside the plateau and the empty level in the tail (Janssen kidney nuclei, 2026-11-16).

    What every 10x library looked at so far has in common is that the plateau is the tallest
    narrow peak of the barcode-count histogram on a log depth axis, standing out of the
    decaying tail on its left. So:
      1. take the tallest peak that rises at least twofold above the lowest point to its left;
      2. measure its width from its lower side, which cells cannot contaminate, and mirror it:
         that is the plateau, with twice as many barcodes as lie below its centre;
      3. whatever the histogram holds above the centre in excess of the mirrored plateau is
         cells. The share of excess at each depth, made monotone, is the prior probability
         that a barcode of that depth holds a cell.
    No shape is assumed for the cells, so a shoulder works as well as a separate hump.

    Returns None when no such peak exists, when it is too broad to be a plateau (a matrix of
    called cells only, split-pool data), or when nothing lies above it. The caller then falls
    back to the order-of-magnitude rule. Otherwise a dict: the plateau level P, its width sd0
    (log10), the lower edge of the plateau (cliff), the depth T above which a barcode is at
    least 90% likely a cell on depth alone, the depth hi below which it is at least 90% likely
    empty, the estimated numbers of empties and cells, and the prior curve (grid_x, grid_f).
    """
    Nv = N[N >= low]
    if Nv.size < 2000:
        return None
    x_all = np.log10(Nv)
    edges = np.arange(np.log10(low), x_all.max() + bin_width, bin_width)
    if edges.size < 14:
        return None
    h, _ = np.histogram(x_all, bins=edges)
    h = h.astype(np.float64)
    mid = 0.5 * (edges[:-1] + edges[1:])
    kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    hs = np.convolve(h, kern, mode="same")
    best = None
    for i in range(3, hs.size - 1):
        if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1]:
            j = int(np.argmin(hs[:i]))
            if hs[i] >= 2.0 * (hs[j] + 1.0) and (best is None or hs[i] > best[0]):
                best = (float(hs[i]), i, j)
    if best is None:
        return None
    _, i, j = best
    w = hs[i - 1 : i + 2]
    mu0 = float((w * mid[i - 1 : i + 2]).sum() / w.sum())
    # Lower side of the peak: from where it has fallen to 2% of its height (but not past the
    # valley, and at most 0.6 dex away) up to the centre.
    lo = i
    while lo > j and hs[lo - 1] >= 0.02 * hs[i] and mid[lo - 1] > mu0 - 0.6:
        lo -= 1
    left = (np.arange(h.size) >= lo) & (mid < mu0)
    n_left = float(h[left].sum())
    if n_left < 100:
        return None
    sd0 = float(np.sqrt((h[left] * (mid[left] - mu0) ** 2).sum() / n_left))
    sd0 = max(sd0, 0.5 * bin_width)
    if sd0 > 0.3:
        return None  # too broad for a plateau: this peak is cells
    n0 = 2.0 * n_left
    up = mid > mu0
    expected = n0 * np.exp(-0.5 * ((mid - mu0) / sd0) ** 2) / (sd0 * np.sqrt(2.0 * np.pi)) * bin_width
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(h > 0, 1.0 - expected / h, np.where(mid > mu0 + 3.0 * sd0, 1.0, 0.0))
    f = np.clip(f, 0.0, 1.0)
    f[~up] = 0.0
    f = np.maximum.accumulate(np.convolve(f, np.array([1.0, 2.0, 1.0]) / 4.0, mode="same") * up)
    n_cells = float((h * f).sum())
    if n_cells < 50 or h[mid > mu0 + 3.0 * sd0].sum() < 50:
        return None  # nothing above the plateau
    sure = np.flatnonzero(f >= 0.9)
    unsure = np.flatnonzero((f <= 0.1) & up)
    T = float(10.0 ** edges[sure[0]]) if sure.size else float(10.0 ** (mu0 + 3.0 * sd0))
    hi = float(10.0 ** edges[unsure[-1] + 1]) if unsure.size else float(10.0**mu0)
    cliff = float(10.0 ** max(edges[lo], mu0 - 3.0 * sd0))
    return {"P": float(10.0**mu0), "mu0": mu0, "sd0": sd0, "cliff": cliff, "T": max(T, hi * 1.01), "hi": max(hi, cliff * 1.01),
            "n_empty": n0, "n_cells": n_cells, "grid_x": mid, "grid_f": f}


def hazard_dips(N: np.ndarray, low: float = 5.0, bin_width: float = 0.05, min_above: int = 50) -> list:
    """Every gap in the depth distribution, as dips of the depth hazard, deepest barcodes first.

    On a log depth axis let h(x) be the number of barcodes in a bin and r(x) the number at or
    above it. The hazard h/r is the share of "everything at least this deep" that sits exactly
    here; the barcode rank curve is steepest where it is lowest. Cell populations and background
    both have a high hazard and a gap between them a low one, whatever the platform: the cliff
    above a droplet plateau (10x), or the knee above a long tail with no plateau at all (BD
    Rhapsody tables, split-pool data). A dip is a local minimum at no more than half the typical
    hazard above it, followed by a recovery of at least 1.5 times below it.

    Two populations of cells of different size (a human and a mouse line) also leave a dip
    between them, so callers take the lowest dip above their floor, not the first one.

    Each dip is a dict: T (its depth), hi (the depth just below it where the hazard has
    recovered), the hazards at and above it, and the number of barcodes above it.
    """
    Nv = N[N >= low]
    if Nv.size < 500:
        return []
    x = np.log10(Nv)
    edges = np.arange(np.log10(low), x.max() + bin_width, bin_width)
    if edges.size < 14:
        return []
    h, _ = np.histogram(x, bins=edges)
    h = h.astype(np.float64)
    mid = 0.5 * (edges[:-1] + edges[1:])
    r = np.cumsum(h[::-1])[::-1]
    kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    lam = np.convolve(h, kern, mode="same") / np.maximum(np.convolve(r, kern, mode="same"), 1.0)
    ok = np.flatnonzero(r >= min_above)
    if ok.size < 8:
        return []
    top = int(ok[-1])  # deepest bin with enough barcodes above it
    # Local minima that are low against what lies above them and are followed by a recovery below.
    found = []
    for j in range(top - 3, 2, -1):
        above = lam[j + 1 : top + 1]
        if above.size < 3 or lam[j] > lam[j - 1] or lam[j] > lam[j + 1]:
            continue
        ref = float(np.median(above))
        if lam[j] <= 0.5 * ref and lam[:j].max() >= 1.5 * max(lam[j], 1e-9):
            found.append((j, ref))
    # Minima with no real rise between them are one gap: keep its deepest point (a shallow dip
    # between two kinds of cells must not hide the gap below them). Ties go to the upper end.
    dips = []
    for j, ref in found:  # deepest barcodes first
        if dips:
            j0 = dips[-1]["bin"]
            if lam[j + 1 : j0].size == 0 or lam[j + 1 : j0].max() < 1.5 * max(lam[j], lam[j0], 1e-9):
                if lam[j] < lam[j0]:
                    dips[-1].update({"bin": j, "lam_knee": float(lam[j]), "lam_above": ref})
                continue
        dips.append({"bin": j, "lam_knee": float(lam[j]), "lam_above": ref})
    for d in dips:
        j = d.pop("bin")
        rec = np.flatnonzero(lam[:j] >= 1.5 * max(lam[j], 1e-9))
        k = int(rec[-1]) if rec.size else 0  # nearest bin below the dip where the hazard has recovered
        d.update({"T": float(10.0 ** mid[j]), "hi": float(10.0 ** edges[k + 1]), "n_above": int(r[j])})
    return dips


def hazard_knee(N: np.ndarray, low: float = 5.0, floor: float = 50.0):
    """The lowest gap at or above floor counts: where cells end and background begins. None if no gap."""
    cand = [d for d in hazard_dips(N, low) if d["T"] >= floor]
    return min(cand, key=lambda d: d["T"]) if cand else None


def _umi_median(n: np.ndarray) -> float:
    """Depth below which half of the UMIs of these barcodes lie."""
    if n.size == 0:
        return 0.0
    o = np.sort(n)
    c = np.cumsum(o)
    return float(o[min(int(np.searchsorted(c, 0.5 * c[-1])), o.size - 1)])


def peak_dispersion(X: sp.csr_matrix, N: np.ndarray, lo: float, hi: float, n_genes: int = 50,
                    max_barcodes: int = 3000, seed: int = 0) -> float:
    """How far the barcodes with depth in [lo, hi] are from being draws of one shared profile.

    Cell-free partitions all sample the same ambient RNA, so for each gene the counts across them
    scatter like a multinomial: the index of dispersion sum((x - N p)^2 / (N p)) / (n - 1) is
    near 1. Cells differ from one another, and the index grows with depth and with how different
    they are. Returns the median index over the n_genes most abundant genes (nan if too few
    barcodes). Used to tell an empty-partition plateau from a population of cells.
    """
    rows = np.flatnonzero((N >= lo) & (N <= hi))
    if rows.size < 100:
        return float("nan")
    if rows.size > max_barcodes:
        rows = np.random.default_rng(seed).choice(rows, max_barcodes, replace=False)
    sub = sp.csr_matrix(X[rows]).astype(np.float64)
    tot = np.asarray(sub.sum(axis=1)).ravel()
    col = np.asarray(sub.sum(axis=0)).ravel()
    genes = np.argsort(-col)[:n_genes]
    genes = genes[col[genes] > 0]
    if genes.size < 5 or tot.sum() <= 0:
        return float("nan")
    p_g = col[genes] / tot.sum()
    dense = np.asarray(sub[:, genes].todense())
    mu = np.outer(tot, p_g)
    d = ((dense - mu) ** 2 / np.maximum(mu, 1e-12)).sum(axis=0) / max(rows.size - 1, 1)
    return float(np.median(d))


def prior_logit(N: np.ndarray, mix: dict, clip: float = 3.9) -> np.ndarray:
    """Log-odds that a barcode holds a cell, from its depth and the fitted plateau."""
    f = np.interp(np.log10(np.maximum(N, 1.0)), mix["grid_x"], mix["grid_f"])
    f = np.clip(f, 1e-6, 1.0 - 1e-6)
    return np.clip(np.log(f) - np.log1p(-f), -clip, clip)


def _codes(labels, n):
    if labels is None:
        return np.zeros(n, dtype=np.int64), ["all"]
    names, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return codes.astype(np.int64), list(names)


def _profile(X: sp.csr_matrix, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros(X.shape[1])
    return np.asarray(X[rows].sum(axis=0)).ravel().astype(np.float64)


def estimate_priors(
    X: sp.csr_matrix,
    unit=None,
    pool=None,
    cell_calls: np.ndarray | None = None,
    expected_cells: int | None = None,
    empty_level: float | None = None,
    low_count: float = 5.0,
    max_barcodes: int = 250_000,
    n_train_empties: int | None = None,
    soup_shrink_umis: float = 2000.0,
    seed: int = 0,
) -> Priors:
    """Pick the analyzed barcodes and compute soup profiles, ambient level and weak cell labels.

    cell_calls: optional boolean mask over barcodes (vendor "filtered" calls), used as a
        weak label, never as ground truth.
    """
    rng = np.random.default_rng(seed)
    n_bc = X.shape[0]
    N = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_sorted = np.sort(N)[::-1]

    # Three ways to place the thresholds, in order of preference.
    #   plateau  a narrow peak of cell-free partitions under the cells (droplets). The peak must
    #            also look cell-free: its barcodes all sample one profile (peak_dispersion near 1;
    #            1.0 to 1.3 on 31 10x and BD libraries, above 4 when the peak was really cells).
    #   knee     no plateau, but a gap between cells and a long background tail (BD Rhapsody
    #            tables, split-pool data): the lowest dip of the depth hazard.
    #   order of magnitude   neither; also used when the caller fixes a threshold or gives cell calls.
    auto = not expected_cells and not empty_level and cell_calls is None
    mix, knee, rejected, level_stat = None, None, None, np.median
    dips = hazard_dips(N, low_count) if auto else []
    if auto:
        mix = plateau_mixture(N, low_count)
        if mix is not None:
            disp = peak_dispersion(X, N, mix["P"] * 10.0 ** (-mix["sd0"]), mix["P"] * 10.0 ** mix["sd0"], seed=seed)
            mix["dispersion"] = disp
            if np.isfinite(disp) and disp > 2.0:
                rejected, mix = {"P": float(mix["P"]), "dispersion": float(disp)}, None
        if mix is None:
            cand = [d for d in dips if d["T"] >= 50.0]
            knee = min(cand, key=lambda d: d["T"]) if cand else None
    if mix is not None:
        T, P, hi = mix["T"], mix["P"], mix["hi"]
        n_cells = int((N >= T).sum())
        bg_floor = plateau_floor = mix["cliff"]
        # The ambient level is read from the whole plateau (two widths either side of its centre),
        # not only from the surely-empty lower part used for the soup profile.
        e_window = (P * 10.0 ** (-2.0 * mix["sd0"]), P * 10.0 ** (2.0 * mix["sd0"]))
        empty_sd0 = float(mix["sd0"])
    elif knee is not None:
        T, hi = knee["T"], min(knee["hi"], knee["T"] / 1.5)
        n_cells = int((N >= T).sum())
        sub = N[(N >= low_count) & (N <= hi)]
        # No plateau to read the ambient level from. The reference level is the depth that splits
        # the background's UMIs in half; the calibration step then scales it (ambient multiplier).
        P = max(_umi_median(sub), low_count)
        bg_floor = plateau_floor = max(low_count, P / 3.0)
        e_window, level_stat = (low_count, hi), _umi_median
        win = sub[sub >= plateau_floor]
        empty_sd0 = float(np.std(np.log10(win))) if win.size > 10 else 0.3
    else:
        T, n_cells = ordmag_threshold(n_sorted, expected_cells, low_count)
        if cell_calls is not None and cell_calls.sum() >= 50:
            T = max(low_count, float(np.quantile(N[cell_calls], 0.01)))
            n_cells = int(cell_calls.sum())
        P = float(empty_level) if empty_level else empty_mode(N, T, low_count)
        hi = max(min(1.5 * P, T / 3.0), low_count + 1.0)
        bg_floor, plateau_floor = low_count, max(low_count, 0.5 * P)
        e_window = (plateau_floor, hi)
        empty_sd0 = None
    # Depth above which a barcode is a cell beyond doubt: the lowest gap above the threshold when
    # there is one (the plateau's own threshold only says "not plateau").
    floor = P * 10.0 ** (2.0 * mix["sd0"]) if mix is not None else T
    gaps = [d["T"] for d in dips if d["T"] >= floor]
    confident = float(max(T, min(gaps))) if gaps else float(T)

    unit_codes, unit_names = _codes(unit, n_bc)
    pool_codes, pool_names = _codes(pool, n_bc)

    # Barcodes the model sees: everything above the background window, plus a sample of
    # plateau barcodes that teach it the ambient level and act as negative examples.
    upper = np.flatnonzero(N > hi)
    if upper.size > max_barcodes:
        upper = upper[np.argsort(-N[upper], kind="stable")[:max_barcodes]]
    plateau = np.flatnonzero((N >= plateau_floor) & (N <= hi))
    n_emp = n_train_empties or int(np.clip(upper.size, 5000, 50000))
    emp = plateau if plateau.size <= n_emp else rng.choice(plateau, n_emp, replace=False)
    analyzed = np.sort(np.concatenate([upper, emp]))

    # Soup profile per unit from the surely-background barcodes, shrunk toward the pooled
    # profile when a unit has few background UMIs. With a plateau these are the plateau
    # partitions only: the low-count tail is mostly mis-read barcodes of real cells, so its
    # profile leans toward the cells rather than the ambient RNA.
    bg = np.flatnonzero((N >= bg_floor) & (N <= hi))
    pooled = _profile(X, bg)
    pooled_p = (pooled + 1e-12) / (pooled.sum() + 1e-12 * pooled.size)
    soup = np.zeros((len(unit_names), X.shape[1]))
    e_unit = np.zeros(len(unit_names))
    soup_umis = {}
    level = np.flatnonzero((N >= e_window[0]) & (N <= e_window[1]))
    e_pooled = float(level_stat(N[level])) if level.size else P
    for u, name in enumerate(unit_names):
        cnt = _profile(X, bg[unit_codes[bg] == u])
        soup[u] = (cnt + soup_shrink_umis * pooled_p) / (cnt.sum() + soup_shrink_umis)
        soup_umis[name] = float(cnt.sum())
        pl = level[unit_codes[level] == u]
        e_unit[u] = float(level_stat(N[pl])) if pl.size >= 50 else e_pooled

    # Average cell profile per pool (the profile of chimeric / swapped reads).
    cells = np.flatnonzero(N >= T) if cell_calls is None else np.flatnonzero(cell_calls)
    pooled_c = _profile(X, cells)
    chibar = np.zeros((len(pool_names), X.shape[1]))
    for p_, _ in enumerate(pool_names):
        cnt = _profile(X, cells[pool_codes[cells] == p_])
        chibar[p_] = cnt if cnt.sum() > 0 else pooled_c

    genes = np.flatnonzero(np.asarray(X[analyzed].sum(axis=0)).ravel() > 0)
    floor = 1e-9 / max(genes.size, 1)
    soup = soup[:, genes] + floor
    soup /= soup.sum(axis=1, keepdims=True)
    chibar = chibar[:, genes] + floor
    chibar /= chibar.sum(axis=1, keepdims=True)

    # Weak prior on cell presence from the count alone: 0.05 at the top of the background
    # window, 0.95 at the cell threshold. Vendor calls, when given, override upward.
    Na = N[analyzed]
    if mix is not None:
        logit = prior_logit(Na, mix)
    else:
        t_mid = np.sqrt(hi * max(T, hi * 1.01))
        tau = max((np.log(max(T, hi * 1.01)) - np.log(hi)) / 6.0, 0.1)
        logit = np.clip((np.log(Na) - np.log(t_mid)) / tau, -3.9, 3.9)
    if cell_calls is not None:
        called = cell_calls[analyzed]
        logit = np.where(called, 3.9, np.minimum(logit, 0.0))

    logN_cells = np.log(N[cells]) if cells.size else np.log(np.array([T * 3.0]))
    med = float(np.median(logN_cells))
    mad = float(np.median(np.abs(logN_cells - med))) * 1.4826
    d_scale = max(1.0, mad)

    diag = {
        "threshold_method": "plateau" if mix is not None else ("knee" if knee is not None else "order of magnitude"),
        "confident_threshold": confident,
        "n_confident": int((N >= confident).sum()),
        "empty_sd0": empty_sd0,
        "knee": {k: round(float(v), 4) for k, v in knee.items()} if knee is not None else None,
        "rejected_plateau": rejected,
        "bead_cliff": float(mix["cliff"]) if mix is not None else None,
        "plateau": {k: round(float(v), 4) for k, v in mix.items() if np.isscalar(v)} if mix is not None else None,
        "cell_threshold": float(T),
        "n_cells_prior": int(n_cells),
        "empty_level": float(P),
        "empty_max": float(hi),
        "n_background_barcodes": int(bg.size),
        "n_plateau_barcodes": int(plateau.size),
        "n_analyzed": int(analyzed.size),
        "n_train_empties": int(emp.size),
        "n_genes": int(genes.size),
        "soup_umis_per_unit": soup_umis,
        "e_unit_init": {n: float(v) for n, v in zip(unit_names, e_unit)},
    }
    return Priors(
        cell_threshold=float(T),
        empty_level=float(P),
        empty_max=float(hi),
        analyzed=analyzed,
        genes=genes,
        prior_cell_logit=logit.astype(np.float32),
        unit=unit_codes[analyzed],
        pool=pool_codes[analyzed],
        unit_names=unit_names,
        pool_names=pool_names,
        soup=soup.astype(np.float32),
        chibar=chibar.astype(np.float32),
        e_unit=np.maximum(e_unit, 1.0).astype(np.float32),
        d_loc=med,
        d_scale=d_scale,
        diagnostics=diag,
    )


def plot_priors(X: sp.csr_matrix, pri: Priors, path: str) -> None:
    """Barcode rank plot with the chosen thresholds, for eyeballing the priors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.sort(np.asarray(X.sum(axis=1)).ravel())[::-1]
    N = N[N > 0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(np.arange(1, N.size + 1), N, color="#16233A", lw=1.2)
    for y, lab, col in (
        (pri.cell_threshold, "cell threshold", "#C2872B"),
        (pri.empty_max, "background window top", "#6E7887"),
        (pri.empty_level, "empty level", "#3A6FB5"),
        (pri.diagnostics.get("confident_threshold", pri.cell_threshold), "surely cells above", "#2F9B70"),
    ):
        ax.axhline(y, color=col, lw=1, ls="--", label=f"{lab} = {y:.0f}")
    ax.set_xlabel("barcode rank")
    ax.set_ylabel("UMI counts")
    ax.set_title(f"thresholds from: {pri.diagnostics.get('threshold_method', '')}", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
