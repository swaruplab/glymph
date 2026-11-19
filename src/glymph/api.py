"""Fit Glymph to one library and write thinned counts.

One library runs on one device. Use `glymph batch` (cli.py) or a Slurm array to put
one library on each GPU of a node.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from ._version import __version__
from .background import Priors, estimate_priors
from .model import GlymphNet
from .sparse import DeviceCSR
from .exclusion import cell_multipliers, silent_budget
from .thinning import apportion, gene_budget, gene_budget_blocked, gene_cell_budget


@dataclass
class Config:
    # model
    n_hidden: int = 512
    n_latent: int = 32
    n_cond: int = 16
    likelihood: str = "multinomial"  # or "dirmult" (overdispersed composition)
    eps_alpha: float = 10.0  # Gamma shape of the per-cell ambient multiplier in the first phase
    eps_alpha_calibrated: float = 50.0  # the same in the second phase when calibration found nothing: held near 1
    eps_alpha_pinned: float = 2000.0  # the same when calibration set per-cell multipliers: the posterior stays on them
    learn_ambient_level: bool = False  # False: the ambient level stays at the plateau read from the depth histogram
    s_init: float = 0.01  # starting value of the swapped / chimeric fraction
    s_max: float = 0.05  # hard cap on it (0.1 until 0.3.0: depth-proportional background of nuclei now has its own ambient-profile term)
    # Second phase: estimate the background budget per cell (ambient multiplier and swapped
    # fraction) from genes that groups of cells do not express (exclusion.py), fix it there and
    # retrain so the decoder gives the background back.
    calibrate_swap: bool = True
    calib_clusters: int = 20
    cell_multipliers: bool = True  # after calibration, a multiplier per cell from its own silent-gene counts
    mask_silent: bool = True  # second phase and output: the decoder may not express a cluster's silent genes (exclusion constraint)
    silent_broad: bool = True  # silent genes also among genes too rare to test one by one (exact Poisson test against the measured level)
    calib_retry: bool = True  # no silent-gene signal: try again with half as many clusters, down to two
    calib_seeds: int = 3  # clusterings tried for the silent-gene estimate; the middle estimate is kept
    calib_embedding: str = "pca"  # what the calibration clusters are built on: "pca" (principal components of the counts: the same in every run) or "latent" (the model's latent space, as up to 0.3.2)
    silent_tree: bool = True  # silent genes decided on a tree: clusters and their sub-clusters (multires.py); False: one resolution, as up to 0.3.2
    tree_children: int = 8  # sub-clusters per cluster, at most
    tree_min_cells: int = 40  # cells per sub-cluster, about
    tree_carriers: float = 0.25  # a sub-cluster overrules its cluster for a gene that this share of its cells carries
    tree_peel: int = 0  # above 0: the first version of the tree, which also sets up to this many expressing sub-clusters aside and tests the rest (lost removal on BD Rhapsody and marker counts on tissue, 2026-11-18)
    splice: bool = False  # use layers["unspliced"] (nuclei): spliced against unspliced molecules as a second negative control (splice.py). Off by default: on the two kidney nuclei libraries it moved tubule-marker removal by 1 to 3 points either way and raised removed over truth by 0.04 (2026-11-18)
    splice_silent: bool = True  # ... to call a gene silent where its molecules are spliced like background, and never where they are not
    splice_output: bool = True  # ... and to update each entry's background share by its own molecules before counts are removed
    swap_contrast: bool = True  # which profile the depth-proportional background has (soup or average cell) is read from gene contrast (exclusion._swap_shape); False: left to the per-cell fit, which cannot tell and kept the swapped fraction at its start (up to 0.3.2)
    calib_cross_check: bool = True  # the background level is fitted under two rules for which gene and cluster pairs are tested (they follow the fit, or are settled at the start), and where the two differ by more than half the lower one is kept: each rule runs away on some library (exclusion.silent_budget). False: one rule only, the next option
    calib_fixed_candidates: bool = False  # the single rule when calib_cross_check is off: False, the tested pairs follow the fit (up to 0.4.0rc1); True, they are settled at the starting amounts
    swap_gate: float = 0.05  # the search is made only where the depth-proportional background is at most this share of depth; beyond it (nuclei) that background keeps the soup profile
    swap_cap: float = 0.05  # upper bound of the swapped fraction read from silent genes; what it takes stays on the soup profile (0.3 was tried on 2026-11-18 and over-cleaned kidney nuclei)
    silent_factor_max: float = 1.0  # above 1: a gene at the same multiple of the fitted background in every cluster that does not express it is silent there, up to this multiple. Off: right on cell-line mixtures (3 gave +2 points on two 10x libraries), wrong on tissue, where such genes are weakly expressed (kidney whole cells: 1.8 times the true background removed, 2026-11-18)
    superpose: bool = False  # a cluster (or sub-cluster) that is a larger cluster plus background-like RNA takes that cluster's silent genes. Off: it cleans the 150 dirty cells of a BD cell-line mixture (+4 points) and merges true cell types in tissue (kidney: up to 2.1 times the true background removed; a xenograft lost 11 points, 2026-11-18)
    factor_budget: bool = True  # ... and that multiple is the gene's background level in every cell (False: only the silent sets use it)
    splice_shift: float = 0.0  # logit shift of the unspliced share from the empty partitions to background inside cells (negative: fitted on the count-silent genes, the first version)
    doublet_match: float = 0.5  # a cell with far more counts on its cluster's silent genes than background explains is left alone when those counts match the background profile less than this (0: never, as in 0.3.1)
    doublet_share_max: float = 0.2  # ... or when more than this share of its counts is on those genes (1: no limit)
    cluster_match: float = 0.0  # ... or when its cluster's own expression explains less than this share of its counts (0: not used, the default; 0.3 and 0.5 were tested for 0.3.2 and changed little)
    spare_doublets: bool = False  # True: every such cell is left alone, whatever its profile
    calib_ridge_max: float = 5.0  # a clustering whose level and fit still disagree by more than this at the end does not count
    match_posterior_mean: bool = False  # after the second phase, shift the multipliers' prior so the posterior mean lands on the calibrated value, and train a third, short phase
    keep_cell_totals: bool = True  # output rule also keeps each cell's expected background total (thinning.gene_cell_budget)
    cell_totals_max_entries: int = 150_000_000  # above this many entries the gene-only rule is used, in blocks
    calib_quantile: float = 0.5  # point of the per-group estimates used (the bound holds for every group)
    calib_steps_frac: float = 0.5  # steps of the second phase as a share of target_steps; it runs them all
    # Experimental, off by default: a parsimony prior that pushes s up until the data stop it.
    # On real mixtures the likelihood pushed s down instead (the background's gene profile
    # differs from both fixed profiles, so the decoder explains it better), and removal fell.
    bg_reward: float = 0.0  # nats per UMI per unit s
    s_freeze_frac: float = 0.0  # share of steps with s held at s_init
    global_lr_mult: float = 1.0  # learning-rate multiplier for the global scalars (e, s)
    # training
    batch_size: int = 1024
    target_steps: int = 3000
    min_epochs: int = 20
    max_epochs: int = 300
    lr: float = 1e-3
    kl_warmup_frac: float = 0.2
    val_frac: float = 0.05
    patience: int = 8
    seed: int = 0
    # priors
    low_count: float = 5.0
    expected_cells: int | None = None
    empty_level: float | None = None
    max_barcodes: int = 250_000
    # output
    estimator: str = "gene"  # gene | budget | sample | map | expected
    # False-positive budget of the gene estimator, the same idea as CellBender's --fpr (whose default is 0.01).
    # 0 removes the expected background only. 0.005 was chosen on the first benchmark (seven 10x human + mouse
    # mixtures), where it gave the same native loss as CellBender; 0.01 matched CellBender's removal but lost more.
    fpr: float = 0.003  # false-positive budget: extra removal allowed per gene as a share of its native counts. 0.3% is where same-species removal equals its expected value on the 15 development mixtures with 0.4.0 (1.00 by the benchmark's extrapolation, 1.02 with the split measured inside cells; 0.4% gave 1.08 and 1.10 once the swapped fraction was read from gene contrast, 2026-11-18). 0.4% was the value for 0.3.0 to 0.3.2.
    fpr_background_cap: float = -1.0  # a gene's false-positive allowance is at most this share of its expected background (negative: no limit, the default; 0.2 was tested for 0.3.2 and under-cleaned BD Rhapsody)
    swap_start: float = 2e-4  # starting value of the swapped fraction in the silent-gene fit (negative: the learned value, as before 0.3.2)
    silent_wide_level: float = -1.0  # the wider silent set is tested against this multiple of the fitted background (negative: the ridge level, the default; 1.0 was tested for 0.3.2 and lost removal on Parse)
    strength: float = 1.0
    cell_prob_cutoff: float = 0.5
    keep_non_cells: bool = False
    device: str = "auto"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Timer:
    def __init__(self, device):
        self.device, self.t0, self.marks = device, time.perf_counter(), {}

    def mark(self, name):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.marks[name] = round(now - self.t0, 3)
        self.t0 = now


def fit(model: GlymphNet, csr: DeviceCSR, unit, pool, prior_logit, cfg: Config, log=print, phase: str = "main") -> dict:
    """phase="main": the full fit. phase="calibrated": s is held fixed at its current value
    (set from the exclusion bound), no KL warm-up, and no early stopping: with s raised the
    decoder has to give the background back, which barely moves the validation objective, so
    a plateau rule stops it after a few epochs with the job half done."""
    calibrated = phase == "calibrated"
    device = csr.device
    n = csr.shape[0]
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=gen).to(device)
    n_val = int(cfg.val_frac * n) if n >= 400 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    bs = min(cfg.batch_size, tr_idx.numel())
    steps_per_epoch = math.ceil(tr_idx.numel() / bs)
    target = cfg.target_steps * (cfg.calib_steps_frac if calibrated else 1.0)
    epochs = int(np.clip(math.ceil(target / steps_per_epoch), cfg.min_epochs, cfg.max_epochs))
    total = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total))
    kl_steps = 1 if calibrated else max(1, int(cfg.kl_warmup_frac * total))

    # Report the objective in nats per UMI so its scale does not depend on depth.
    scale = 1.0 / float(csr.data.sum() / n)
    glob = model.global_parameters()
    glob_ids = {id(t) for t in glob}
    opt = torch.optim.Adam([
        {"params": [t for t in model.parameters() if id(t) not in glob_ids], "lr": cfg.lr},
        {"params": glob, "lr": cfg.lr * cfg.global_lr_mult},
    ])
    s_free_at = 0 if calibrated else int(cfg.s_freeze_frac * total)
    model.logit_s.requires_grad_(s_free_at == 0 and not calibrated)

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def evaluate(idx):
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, idx.numel(), 4096):
                b = idx[i : i + 4096]
                x, nz = csr.dense(b)
                out = model(x, nz, unit[b], pool[b], prior_logit[b], 1.0, sample=False, log_kappa=model.lk(b), cluster=model.sc(b))
                tot += float(out.elbo.sum())
        # Same objective as training: the parsimony prior on s is part of it.
        return -tot * scale / max(1, idx.numel()) - cfg.bg_reward * float(model.s().detach().mean())

    best, best_state, bad, step, history, s_hist = float("inf"), None, 0, 0, [], []
    for ep in range(epochs):
        model.train()
        order = tr_idx[torch.randperm(tr_idx.numel(), generator=gen).to(device)]
        run = 0.0
        for i in range(0, order.numel(), bs):
            b = order[i : i + bs]
            x, nz = csr.dense(b)
            out = model(x, nz, unit[b], pool[b], prior_logit[b], min(1.0, step / kl_steps), log_kappa=model.lk(b), cluster=model.sc(b))
            if step == s_free_at and not calibrated:
                model.logit_s.requires_grad_(True)
            loss = (-out.elbo.mean() - model.global_logprior() / n) * scale - cfg.bg_reward * model.s().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss.detach())
        tr = run / steps_per_epoch
        va = evaluate(val_idx) if n_val else tr
        s_now = float(model.s().detach().mean())
        s_hist.append(s_now)
        history.append({"epoch": ep + 1, "train": round(tr, 5), "val": round(va, 5), "s": round(s_now, 5)})
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"[glymph] epoch {ep + 1}/{epochs}  train {tr:.4f}  val {va:.4f}  s {s_now:.4f}")
        # s is still moving if it changed by more than 2% over the last `patience` epochs.
        k = cfg.patience
        s_moving = cfg.bg_reward > 0 and not calibrated and (
            step < s_free_at or len(s_hist) <= k or abs(s_hist[-1] - s_hist[-1 - k]) > 0.02 * max(s_hist[-1], 1e-6)
        )
        if calibrated:
            best = min(best, va)
        elif step >= max(kl_steps, s_free_at):  # compare epochs once the KL weight is 1 and s is free
            if va < best - 1e-5 or s_moving:
                best, bad = min(best, va), 0
                best_state = {k_: t.detach().clone() for k_, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"[glymph] early stop at epoch {ep + 1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"epochs_run": len(history), "epochs_planned": epochs, "steps": step, "best_val": best, "history": history}


@torch.no_grad()
def calibrate_background(model: GlymphNet, csr: DeviceCSR, unit, pool, prior_logit, cell_threshold: float, cfg: Config, u=None) -> dict:
    """Set the model's ambient multiplier and swapped fraction from silent genes (exclusion.silent_budget).

    Uses barcodes the model calls cells that are also at or above the confident-cell depth,
    clustered on the latent space. A pool without an estimate keeps kappa = 1 and its learned
    swapped fraction.
    """
    device, n = csr.device, csr.shape[0]
    z = torch.zeros(n, model.encoder.z.out_features // 2, device=device)
    prob = torch.zeros(n, device=device)
    tot = torch.zeros(n, device=device)
    for i in range(0, n, 4096):
        b = torch.arange(i, min(i + 4096, n), device=device)
        x, nz = csr.dense(b)
        post = model.posterior(x, nz, unit[b], pool[b], prior_logit[b], model.lk(b), model.sc(b))
        z[b], prob[b], tot[b] = post.z, post.cell_prob, post.N
    rows = torch.nonzero((prob >= cfg.cell_prob_cutoff) & (tot >= cell_threshold), as_tuple=True)[0]
    if cfg.calib_embedding == "pca" and rows.numel() >= 50:
        from .multires import pca_embedding

        # Clusters from the counts themselves: the latent space differs between two runs on a GPU, and on
        # libraries with few testable genes the estimate moved with it.
        z = pca_embedding(csr, rows, torch.arange(n, device=device), seed=cfg.seed)
    learned = model.s().detach().clone()
    e_now, soup_now, chibar_now = model.e().detach(), torch.exp(model.log_soup), torch.exp(model.log_chibar)
    splice = None
    if u is not None and cfg.splice and cfg.silent_tree:
        bg = torch.nonzero(prob < cfg.cell_prob_cutoff, as_tuple=True)[0]
        splice = {"u": u, "bg_rows": bg, "bg_pool": pool[bg], "silent": cfg.splice_silent, "shift": cfg.splice_shift}
    kappa, swap, diag, states = silent_budget(
        csr, rows, z[rows], tot[rows], unit[rows], pool[rows], e_now, soup_now, chibar_now, learned, splice=splice,
        n_clusters=cfg.calib_clusters, s_max=max(model.s_max, cfg.swap_cap) if cfg.swap_contrast else model.s_max, seed=cfg.seed, broad=cfg.silent_broad, retry=cfg.calib_retry, seeds=cfg.calib_seeds, s_start=cfg.swap_start, wide_level=cfg.silent_wide_level, ridge_max=cfg.calib_ridge_max,
        tree=cfg.silent_tree, tree_children=cfg.tree_children, tree_min_cells=cfg.tree_min_cells, tree_peel=cfg.tree_peel, tree_carriers=cfg.tree_carriers,
        swap_contrast=cfg.swap_contrast, swap_gate=cfg.swap_gate, fixed_candidates=cfg.calib_fixed_candidates, cross_check=cfg.calib_cross_check, factor_max=cfg.silent_factor_max,
        superpose=cfg.superpose, superpose_share=cfg.doublet_share_max if cfg.doublet_share_max < 1 else 0.2, superpose_match=cfg.doublet_match if cfg.doublet_match > 0 else 0.5,
    )
    model.log_phi = None
    new_s, new_k = learned.clone(), torch.ones_like(learned)
    for p_ in range(len(kappa)):
        if kappa[p_] is not None:
            new_k[p_], new_s[p_] = kappa[p_], swap[p_]
    if float(new_s.max()) > 0.8 * model.s_max:  # a swapped fraction read above the bound of the learned one (nuclei): the bound moves
        model.s_max = float(1.25 * new_s.max())
    ratio = (new_s / model.s_max).clamp(1e-4, 1.0 - 1e-4)
    model.logit_s.copy_(torch.log(ratio / (1.0 - ratio)))
    model.log_kappa.copy_(torch.log(new_k))
    # Every barcode then gets its own multiplier: the fitted trend in depth and, for called cells
    # (cfg.cell_multipliers), the evidence of its own counts on its cluster's silent genes.
    per_cell = {}
    kappa_cell = new_k[pool].clone()
    excess = torch.zeros(n, dtype=torch.bool, device=device)
    bg_match = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    cl_match = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    sil_share = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    for p_, st in enumerate(states):
        if st is None:
            continue
        sel = torch.nonzero(pool == p_, as_tuple=True)[0]
        called = (prob[sel] >= cfg.cell_prob_cutoff) if cfg.cell_multipliers else torch.zeros(sel.numel(), dtype=torch.bool, device=device)
        soup_p, chibar_p = soup_now, chibar_now[p_]
        phi = st.get("phi")
        if phi is not None and bool((phi != 1.0).any()):  # genes with a background level of their own (exclusion: shared level)
            soup_p, chibar_p = soup_now * phi.to(soup_now.dtype).unsqueeze(0), chibar_p * phi.to(chibar_p.dtype)
            if cfg.factor_budget:
                if model.log_phi is None:
                    model.log_phi = torch.zeros_like(model.log_chibar)
                model.log_phi[p_] = torch.log(phi.to(model.log_chibar.dtype))
        m, per_cell[str(p_)], exc, mt = cell_multipliers(csr, sel, z[sel], tot[sel], unit[sel], e_now, soup_p, chibar_p, st, called, cluster_test=cfg.cluster_match > 0)
        kappa_cell[sel] = m
        excess[sel] = exc
        bg_match[sel], cl_match[sel], sil_share[sel] = mt[0].to(torch.float32), mt[1].to(torch.float32), mt[2].to(torch.float32)
    # Cells with far more counts on their cluster's silent genes than background explains. A heavily
    # contaminated cell stays held to the silent genes: its counts there follow the background
    # profile and are at most a fifth of the cell. A cell that fails either condition carries
    # expression of its own on those genes (a doublet, or a cell type too rare for its own cluster)
    # and is left alone (see exclusion.cell_multipliers and background_match).
    if cfg.spare_doublets:
        spared = excess.clone()
    else:
        spared = torch.zeros_like(excess)
        if cfg.doublet_match > 0:
            spared |= excess & ~(bg_match >= cfg.doublet_match)
        if cfg.doublet_share_max < 1:
            spared |= excess & (sil_share > cfg.doublet_share_max)
        if cfg.cluster_match > 0:
            spared |= excess & ~(cl_match >= cfg.cluster_match)
    sparing = cfg.spare_doublets or cfg.doublet_match > 0 or cfg.doublet_share_max < 1 or cfg.cluster_match > 0
    doublet_like = spared if sparing else excess
    if per_cell:
        model.cell_log_kappa = torch.log(kappa_cell.clamp_min(1e-3))
    # Exclusion constraint (cfg.mask_silent): hand the model each cluster's silent genes.
    if cfg.mask_silent and any(st is not None for st in states):
        rows_, offset = [], 0
        cluster = torch.full((n,), -1, dtype=torch.long, device=device)
        for p_, st in enumerate(states):
            if st is None:
                continue
            sel = torch.nonzero((pool == p_) & (prob >= cfg.cell_prob_cutoff) & ~spared, as_tuple=True)[0]
            if sel.numel():
                cluster[sel] = torch.cdist(z[sel].double(), st["centres"]).argmin(1) + offset
            rows_.append(st.get("mask_strip", st.get("mask_broad", st["mask"])))
            offset += st["mask"].shape[0]
        mask = torch.cat(rows_ + [torch.zeros(1, rows_[0].shape[1], dtype=torch.bool, device=device)]).to(torch.float32)
        cluster[cluster < 0] = offset  # the empty last row
        model.silent_mask, model.cell_cluster = mask, cluster
        per_cell["masked_cells"] = int((cluster < offset).sum())
        per_cell["silent_excess_cells"] = int(excess.sum())
        per_cell["doublet_like_cells"] = int(spared.sum())
        per_cell["silent_genes_per_cluster_median"] = float(torch.median(mask[:-1].sum(1)))
    model.doublet_like, model.background_match, model.cluster_match, model.silent_share = doublet_like, bg_match, cl_match, sil_share
    # per-gene unspliced shares of background and of own expression, one row per pool, for the output step
    model.splice_prof = None
    if any(st is not None and "splice" in st for st in states):
        G = csr.shape[1]
        prof = {"pi_bg": torch.zeros(len(states), G, device=device), "pi_nat": torch.zeros(len(states), G, device=device),
                "informative": torch.zeros(len(states), G, dtype=torch.bool, device=device)}
        for p_, st in enumerate(states):
            if st is not None and "splice" in st:
                for key in prof:
                    prof[key][p_] = st["splice"][key].to(prof[key].dtype)
        model.splice_prof = prof
    return {"applied": any(v is not None for v in kappa), "per_cell": per_cell,
            "swap_learned": [float(v) for v in learned.cpu()], "swap_estimate": swap, "kappa_estimate": kappa,
            "kappa_used": [float(v) for v in new_k.cpu()], "swap_used": [float(v) for v in model.s().detach().cpu()],
            "n_confident_cells": int(rows.numel()), "confident_threshold": float(cell_threshold), "pools": diag}


@torch.no_grad()
def posterior_drift(model: GlymphNet, csr: DeviceCSR, unit, pool, prior_logit, cell_threshold: float, cfg: Config) -> torch.Tensor:
    """Per pool: summed posterior ambient multiplier of the confident cells over the summed prior mean."""
    device, n = csr.device, csr.shape[0]
    n_pools = model.log_kappa.numel()
    num = torch.zeros(n_pools, device=device, dtype=torch.float64)
    den = torch.zeros(n_pools, device=device, dtype=torch.float64)
    for i in range(0, n, 4096):
        b = torch.arange(i, min(i + 4096, n), device=device)
        x, nz = csr.dense(b)
        post = model.posterior(x, nz, unit[b], pool[b], prior_logit[b], model.lk(b), model.sc(b))
        ok = ((post.cell_prob >= cfg.cell_prob_cutoff) & (post.N >= cell_threshold)).double()
        prior = torch.exp(model.lk(b)) if model.cell_log_kappa is not None else torch.exp(model.log_kappa[pool[b]])
        num.index_add_(0, pool[b], ok * post.eps.double())
        den.index_add_(0, pool[b], ok * prior.double())
    return (num / den.clamp_min(1e-9)).clamp(0.5, 1.5).to(torch.float32)


@torch.no_grad()
def denoise(model: GlymphNet, csr: DeviceCSR, unit, pool, prior_logit, cfg: Config, cell_threshold: float = 0.0, u=None) -> dict:
    """Posterior pass and thinning. Returns the new data vector in the CSR's own order."""
    device, n = csr.device, csr.shape[0]
    new_data = torch.zeros_like(csr.data)
    share_all = torch.zeros_like(csr.data) if cfg.estimator == "gene" else None
    cols = {k: torch.zeros(n, device=device) for k in ("cell_prob", "eps", "removed", "exp_amb", "exp_swp")}
    totals = torch.zeros(n, device=device)
    z_all = None
    sprof = getattr(model, "splice_prof", None) if (u is not None and cfg.splice and cfg.splice_output) else None
    moved = [0.0, 0.0, 0.0]
    if sprof is not None:
        from .splice import splice_update
    for i in range(0, n, 4096):
        b = torch.arange(i, min(i + 4096, n), device=device)
        x, nz = csr.dense(b)
        r, c_, v, pos = nz
        post = model.posterior(x, nz, unit[b], pool[b], prior_logit[b], model.lk(b), model.sc(b))
        share = post.share_amb + post.share_swp
        if sprof is not None:
            # each entry's background share, updated by how many of its molecules are unspliced (splice.py)
            pp = pool[b][r]
            new = splice_update(share, v, u[pos], sprof["pi_bg"][pp, c_], sprof["pi_nat"][pp, c_], sprof["informative"][pp, c_])
            called = (post.cell_prob >= cfg.cell_prob_cutoff)[r]
            moved[0] += float((v * share)[called].sum())
            moved[1] += float((v * new)[called].sum())
            moved[2] += float((v * (new - share).abs())[called].sum())
            share = new
        zeros = torch.zeros(b.numel(), device=device)
        if share_all is None:
            m = apportion(r, v, share, b.numel(), cfg.estimator, cfg.strength)
            is_cell = post.cell_prob >= cfg.cell_prob_cutoff
            m = torch.where(is_cell[r], m, v)  # a non-cell barcode is all background
            new_data[pos] = v - m
            cols["removed"][b] = zeros.index_add(0, r, m)
        else:
            share_all[pos] = share
        totals[b] = zeros.index_add(0, r, v)
        cols["cell_prob"][b] = post.cell_prob
        cols["eps"][b] = post.eps
        cols["exp_amb"][b] = zeros.index_add(0, r, v * post.share_amb)
        cols["exp_swp"][b] = zeros.index_add(0, r, v * post.share_swp)
        if z_all is None:
            z_all = torch.zeros(n, post.z.shape[1], device=device)
        z_all[b] = post.z
    if share_all is not None:
        # The gene budget is shared across cells, so it is allocated over the whole library at once,
        # but separately for barcodes at or above the cell count threshold and for the marginal
        # ones below it. Marginal barcodes are mostly background, their entries win the allocation,
        # and in tissue there can be tens of thousands of them: sharing one budget let them take
        # removal away from the real cells.
        lens = csr.indptr[1:] - csr.indptr[:-1]
        is_cell = cols["cell_prob"] >= cfg.cell_prob_cutoff
        removed = torch.zeros(n, device=device)
        groups = (is_cell & (totals >= cell_threshold), is_cell & (totals < cell_threshold))
        cap = cfg.fpr_background_cap if cfg.fpr_background_cap >= 0 else None
        if csr.data.numel() <= 2 * cfg.cell_totals_max_entries:
            row = torch.repeat_interleave(torch.arange(n, device=device), lens)
            for group in groups:
                idx = torch.nonzero(group[row], as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                v = csr.data[idx]
                g = csr.indices[idx].long()
                if cfg.keep_cell_totals and idx.numel() <= cfg.cell_totals_max_entries:
                    m = gene_cell_budget(g, row[idx], v, share_all[idx], csr.shape[1], n, cfg.strength, cfg.fpr, cap=cap)
                else:
                    m = gene_budget_blocked(g, v, share_all[idx], csr.shape[1], cfg.strength, cfg.fpr, cap=cap)
                new_data[idx] = v - m  # entries of non-cell barcodes stay at zero: all background
                removed.index_add_(0, row[idx], m)
        else:
            # A large library (more than twice cell_totals_max_entries entries): the per-gene rule, a block
            # of genes at a time, without ever building an index over all entries. Each gene's
            # budget is independent of the others, so the blocks give the same result as one pass.
            code = torch.zeros(n, dtype=torch.uint8, device=device)
            code[groups[0]] = 1
            code[groups[1]] = 2
            entry_code = torch.repeat_interleave(code, lens)
            del code
            edges, acc = [0], 0
            for j, cnt in enumerate(csr.col_nnz.tolist()):
                if acc > 0 and acc + cnt > 20_000_000:
                    edges.append(j)
                    acc = 0
                acc += cnt
            edges.append(csr.shape[1])
            for g0, g1 in zip(edges[:-1], edges[1:]):
                in_block = (csr.indices >= g0) & (csr.indices < g1)
                for which in (1, 2):
                    idx = torch.nonzero(in_block & (entry_code == which), as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    v, g, sh = csr.data[idx], csr.indices[idx].long() - g0, share_all[idx]
                    try:
                        m = gene_budget(g, v, sh, g1 - g0, cfg.strength, cfg.fpr, cap=cap)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        m = gene_budget(g.cpu(), v.cpu(), sh.cpu(), g1 - g0, cfg.strength, cfg.fpr, cap=cap).to(device)
                    new_data[idx] = v - m
                    removed.index_add_(0, torch.searchsorted(csr.indptr, idx, right=True) - 1, m)
                    del idx, v, g, sh, m
                del in_block
        cols["removed"] = torch.where(is_cell, removed, totals)
    out = {k: t.cpu().numpy() for k, t in cols.items()}
    dl = getattr(model, "doublet_like", None)
    out["doublet_like"] = dl.cpu().numpy() if dl is not None else np.zeros(n, dtype=bool)
    bm = getattr(model, "background_match", None)
    out["background_match"] = bm.cpu().numpy() if bm is not None else np.full(n, np.nan, dtype=np.float32)
    cm = getattr(model, "cluster_match", None)
    out["cluster_match"] = cm.cpu().numpy() if cm is not None else np.full(n, np.nan, dtype=np.float32)
    ss = getattr(model, "silent_share", None)
    out["silent_share"] = ss.cpu().numpy() if ss is not None else np.full(n, np.nan, dtype=np.float32)
    out["data"] = new_data.cpu().numpy()
    out["z"] = z_all.cpu().numpy()
    if sprof is not None:
        out["splice_output"] = {"expected_background_counts_before": moved[0], "expected_background_counts_after": moved[1],
                                "counts_moved_between_entries": moved[2]}
    return out


def _plain(o):
    """Make a nested structure JSON and h5ad friendly."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if o is None:
        return "none"
    return o


def remove_background(
    adata: ad.AnnData,
    unit_key: str | None = None,
    pool_key: str | None = None,
    cell_calls=None,
    config: Config | None = None,
    log=print,
):
    """Thin the background out of one raw library.

    adata: raw, unfiltered counts (barcodes x genes).
    unit_key / pool_key: obs columns naming each barcode's soup unit and amplification pool.
    cell_calls: optional iterable of barcodes the vendor pipeline called as cells.
    Returns (AnnData of denoised cells with layers['raw'], report dict).
    """
    cfg = config or Config()
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    timer = _Timer(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.empty(1, device=device)  # create the CUDA context; the stats call fails without one
        torch.cuda.reset_peak_memory_stats(device)

    X = sp.csr_matrix(adata.X)
    unit = adata.obs[unit_key].to_numpy() if unit_key else None
    pool = adata.obs[pool_key].to_numpy() if pool_key else None
    calls = adata.obs_names.isin(np.asarray(cell_calls).astype(str)) if cell_calls is not None else None
    pri: Priors = estimate_priors(
        X, unit, pool, calls, cfg.expected_cells, cfg.empty_level, cfg.low_count, cfg.max_barcodes, seed=cfg.seed
    )
    log(f"[glymph] priors: {pri.diagnostics}")
    timer.mark("priors_s")

    Xa = X[pri.analyzed][:, pri.genes].tocsr()
    Xa.sort_indices()
    if os.environ.get("GLYMPH_DEBUG_DUMP"):  # development only (see exclusion.silent_budget)
        np.savez(os.environ["GLYMPH_DEBUG_DUMP"] + ".index.npz", genes=pri.genes, analyzed=pri.analyzed)
    csr = DeviceCSR(Xa, device)
    u_t = None
    if cfg.splice and "unspliced" in adata.layers:
        # unspliced molecules of every nonzero entry, in the order of the entries (never more than the entry itself)
        Ul = sp.csr_matrix(adata.layers["unspliced"])[pri.analyzed][:, pri.genes].tocsr()
        pat = Xa.copy()
        pat.data = np.ones_like(pat.data)
        Z = (pat + Ul.multiply(pat)).tocsr()
        Z.sort_indices()
        if Z.nnz == Xa.nnz:
            u_t = torch.as_tensor(np.minimum(Z.data - 1, Xa.data).astype(np.float32), device=device)
            log(f"[glymph] unspliced layer: {float(u_t.sum()) / max(float(Xa.data.sum()), 1.0):.3f} of the molecules of the analysed barcodes")
        del Ul, pat, Z
    unit_t = torch.as_tensor(pri.unit, device=device)
    pool_t = torch.as_tensor(pri.pool, device=device)
    prior_t = torch.as_tensor(pri.prior_cell_logit, device=device)
    model = GlymphNet(
        torch.as_tensor(pri.soup),
        torch.as_tensor(pri.chibar),
        torch.as_tensor(pri.e_unit),
        pri.d_loc,
        pri.d_scale,
        cfg.n_hidden,
        cfg.n_latent,
        cfg.n_cond,
        cfg.eps_alpha,
        s_init=cfg.s_init,
        s_max=cfg.s_max,
        likelihood=cfg.likelihood,
    ).to(device)
    # An empty partition's depth is modelled with the spread the plateau actually has.
    sd0 = pri.diagnostics.get("empty_sd0")
    if sd0:
        cv2 = math.expm1((math.log(10.0) * sd0) ** 2)
        model.empty_alpha = float(np.clip(1.0 / max(cv2, 1e-6), 2.0, 50.0))
    model.log_e.requires_grad_(cfg.learn_ambient_level)
    timer.mark("setup_s")

    train = fit(model, csr, unit_t, pool_t, prior_t, cfg, log)
    timer.mark("train_s")
    calib = {"applied": False}
    t_cell = float(pri.diagnostics.get("confident_threshold", pri.diagnostics["cell_threshold"]))
    if cfg.calibrate_swap:
        calib = calibrate_background(model, csr, unit_t, pool_t, prior_t, t_cell, cfg, u=u_t)
        fmt = lambda xs: [None if v is None else round(v, 4) for v in xs]  # noqa: E731
        log(f"[glymph] background budget from silent genes: ambient multiplier {fmt(calib['kappa_estimate'])}, "
            f"swapped fraction {fmt(calib['swap_estimate'])} (learned {fmt(calib['swap_learned'])})")
        # The second phase runs either way: with or without an estimate, the per-cell ambient
        # multiplier is now held near its anchor so the decoder cannot absorb the background.
        # With per-cell multipliers from silent genes the posterior is pinned to them. Left loose
        # (shape 50) it settled 18 to 32% below the calibrated value on every kidney library, the
        # same drift that calibration exists to prevent (decoder absorbing background).
        model.eps_alpha = cfg.eps_alpha_pinned if calib.get("per_cell") else cfg.eps_alpha_calibrated
        train["calibrated"] = fit(model, csr, unit_t, pool_t, prior_t, cfg, log, phase="calibrated")
        if cfg.match_posterior_mean and calib.get("per_cell") and model.cell_log_kappa is not None:
            # A loose prior keeps what the likelihood knows about each cell's ambient load (in nuclei,
            # with a third of the counts ambient, that is a lot), but the posterior settles below the
            # calibrated level. Measure by how much, move the prior up by that factor, train again.
            drift = posterior_drift(model, csr, unit_t, pool_t, prior_t, t_cell, cfg)
            calib["posterior_over_calibrated"] = [float(v) for v in drift.cpu()]
            model.cell_log_kappa = model.cell_log_kappa - torch.log(drift)[pool_t]
            log(f"[glymph] posterior multiplier over calibrated value: {[round(float(v), 3) for v in drift.cpu()]}; prior shifted, third phase")
            train["matched"] = fit(model, csr, unit_t, pool_t, prior_t, cfg, log, phase="calibrated")
            calib["posterior_over_calibrated_after"] = [float(v) for v in (posterior_drift(model, csr, unit_t, pool_t, prior_t, t_cell, cfg) * drift).cpu()]
    timer.mark("calibrate_s")
    res = denoise(model, csr, unit_t, pool_t, prior_t, cfg, t_cell, u=u_t)
    timer.mark("denoise_s")

    n_a = Xa.shape[0]
    dtype = np.float32 if cfg.estimator == "expected" else np.int32
    den = sp.csr_matrix((res["data"].astype(dtype), pri.genes[Xa.indices], Xa.indptr), shape=(n_a, X.shape[1]))
    den.eliminate_zeros()
    raw = X[pri.analyzed]
    n_raw = np.asarray(raw.sum(axis=1)).ravel()
    obs = adata.obs.iloc[pri.analyzed].copy()
    obs["glymph_cell_probability"] = res["cell_prob"]
    obs["glymph_is_cell"] = res["cell_prob"] >= cfg.cell_prob_cutoff
    obs["glymph_raw_counts"] = n_raw
    obs["glymph_removed_counts"] = res["removed"]
    obs["glymph_background_fraction"] = res["removed"] / np.maximum(n_raw, 1)
    obs["glymph_expected_ambient"] = res["exp_amb"]
    obs["glymph_expected_swapped"] = res["exp_swp"]
    obs["glymph_epsilon"] = res["eps"]
    obs["glymph_doublet_like"] = res["doublet_like"]  # a second expression program on the cluster's silent genes (doublet or rare cell type); left alone
    obs["glymph_background_match"] = res["background_match"]  # cells with far more silent-gene counts than background explains: how closely those counts follow the background profile (NaN elsewhere)
    obs["glymph_silent_share"] = res["silent_share"]  # the same cells: share of their counts on their cluster's silent genes (NaN elsewhere)
    if cfg.cluster_match > 0:
        obs["glymph_cluster_match"] = res["cluster_match"]  # the same cells: share of their counts that their cluster's own expression explains
    keep = np.ones(n_a, bool) if cfg.keep_non_cells else obs["glymph_is_cell"].to_numpy()

    var = adata.var.copy()
    soup_full = np.zeros((len(pri.unit_names), X.shape[1]), dtype=np.float32)
    soup_full[:, pri.genes] = pri.soup
    var["glymph_soup_mean"] = soup_full.mean(axis=0)
    out = ad.AnnData(X=den[keep], obs=obs.loc[keep], var=var, layers={"raw": raw[keep].astype(np.int32)})
    out.obsm["X_glymph"] = res["z"][keep]
    timer.mark("assemble_s")

    is_cell = obs["glymph_is_cell"].to_numpy()
    report = {
        "version": __version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if device.type == "cuda" else 0.0,
        "timings": timer.marks,
        "n_cells": int(is_cell.sum()),
        "e_unit": {n: float(v) for n, v in zip(pri.unit_names, model.e().detach().cpu().numpy())},
        "swap_fraction": {n: float(v) for n, v in zip(pri.pool_names, model.s().detach().cpu().numpy())},
        "ambient_multiplier": {n: float(v) for n, v in zip(pri.pool_names, torch.exp(model.log_kappa).cpu().numpy())},
        "empty_alpha": float(model.empty_alpha),
        "swap_calibration": calib,
        "splice_output": res.get("splice_output"),
        "removed_fraction_in_cells": float(res["removed"][is_cell].sum() / max(1.0, n_raw[is_cell].sum())),
        "priors": pri.diagnostics,
        "train": {k: v for k, v in train.items() if k not in ("history", "calibrated", "matched")},
        "train_calibrated": {k: v for k, v in train.get("calibrated", {}).items() if k != "history"},
        "train_matched": {k: v for k, v in train.get("matched", {}).items() if k != "history"},
        "history": train["history"],
        "history_calibrated": train.get("calibrated", {}).get("history", []),
        "config": asdict(cfg),
    }
    report = _plain(report)
    # The h5ad keeps the report as one JSON string: nested lists of mixed types do not survive
    # AnnData's writer, and the full report is also written beside the output by the CLI.
    slim = {k: v for k, v in report.items() if not k.startswith("history")}
    out.uns["glymph"] = {
        "version": __version__,
        "n_cells": report["n_cells"],
        "removed_fraction_in_cells": report["removed_fraction_in_cells"],
        "report_json": json.dumps(slim),
    }
    return out, report
