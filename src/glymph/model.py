"""The Glymph generative model and its amortized guide.

For barcode n with observed counts x[n, :] summing to N[n], background unit u and
amplification pool p:

  empty partition (y = 0)
      composition   x | N ~ Multinomial(N, (1 - s_p) * soup_u + s_p * chibar_p)
      total         N ~ NegBinomial(mean = e_u / (1 - s_p), shape = alpha0)
                    (alpha0 is set from the measured width of the empty-partition plateau)

  cell-containing partition (y = 1)
      ambient UMIs  eps[n] * e_u            profile soup_u     (constant per partition;
                    eps ~ Gamma(alpha, alpha / kappa_p), mean kappa_p, 1 until calibrated)
      swapped UMIs  s_p * N[n]              profile chibar_p   (proportional to depth)
      native UMIs   d[n] = N[n] * (1 - s_p) - eps[n] * e_u     profile chi[n] = decoder(z[n])
      composition   x | N ~ Multinomial(N, w_nat * chi + w_amb * soup_u + w_swap * chibar_p)
      total         d ~ LogNormal(d_loc, d_scale)

soup_u and chibar_p are closed-form estimates passed in as buffers (see background.py).
e_u and s_p are learned point estimates. z, eps and y are amortized by the encoder; y is
enumerated exactly.

The swapped fraction s_p is capped at s_max. An optional parsimony prior on it
(bg_reward in api.Config: log p(s) = bg_reward * (total UMIs) * s) is off by default. The
reasoning behind it was that the likelihood bounds s_p from above (native expression cannot
be negative) but not from below (the decoder can put a little native mass on a cell's silent
genes), so s_p should be pushed up until the bound stops it. On real human + mouse mixtures
that failed: the gene profile of the background inside cells differs from both fixed profiles
by about 0.08 nats per count, so the decoder explains those counts better and the likelihood
pushes s_p down. The fix that worked is in the output stage instead (thinning.gene_budget).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2.0 * math.pi)


class _Encoder(nn.Module):
    def __init__(self, n_genes, n_hidden, n_latent, n_cond):
        super().__init__()
        self.l1 = nn.Linear(n_genes, n_hidden)
        self.n1 = nn.LayerNorm(n_hidden)
        self.film = nn.Linear(n_cond, 2 * n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden // 2)
        self.n2 = nn.LayerNorm(n_hidden // 2)
        nh = n_hidden // 2 + 2 + n_cond
        self.z = nn.Linear(nh, 2 * n_latent)
        self.eps = nn.Linear(nh, 2)
        self.y = nn.Linear(nh, 1)
        for lin in (self.y, self.eps, self.film):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.eps.bias.copy_(torch.tensor([0.0, math.log(0.2)]))

    def forward(self, x, feats, cond):
        h = self.n1(self.l1(torch.log1p(x)))
        g, b = self.film(cond).chunk(2, dim=-1)
        h = F.gelu(h * (1.0 + g) + b)
        h = F.gelu(self.n2(self.l2(h)))
        h = torch.cat([h, feats, cond], dim=-1)
        z_mu, z_lv = self.z(h).chunk(2, dim=-1)
        eps_m, eps_ls = self.eps(h).unbind(dim=-1)
        return z_mu, z_lv.clamp(-10.0, 5.0), eps_m.clamp(-4.0, 4.0), eps_ls.clamp(-5.0, 0.5), self.y(h).squeeze(-1)


class _Decoder(nn.Module):
    def __init__(self, n_genes, n_hidden, n_latent, n_cond, init_log_profile):
        super().__init__()
        self.l1 = nn.Linear(n_latent + n_cond, n_hidden)
        self.n1 = nn.LayerNorm(n_hidden)
        self.out = nn.Linear(n_hidden, n_genes)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.copy_(init_log_profile)  # start at the average cell profile

    def forward(self, z, cond):
        h = F.gelu(self.n1(self.l1(torch.cat([z, cond], dim=-1))))
        return F.log_softmax(self.out(h), dim=-1)


class GlymphNet(nn.Module):
    def __init__(
        self,
        soup: torch.Tensor,  # [U, G]
        chibar: torch.Tensor,  # [P, G]
        e_init: torch.Tensor,  # [U]
        d_loc: float,
        d_scale: float,
        n_hidden: int = 512,
        n_latent: int = 32,
        n_cond: int = 16,
        eps_alpha: float = 10.0,
        s_init: float = 0.01,
        s_max: float = 0.1,
        e_prior_scale: float = 1.0,
        likelihood: str = "multinomial",
    ):
        super().__init__()
        if likelihood not in ("multinomial", "dirmult"):
            raise ValueError(likelihood)
        n_genes = soup.shape[1]
        self.likelihood = likelihood
        self.eps_alpha = float(eps_alpha)  # Gamma shape of the per-cell ambient multiplier (prior mean kappa)
        self.empty_alpha = float(eps_alpha)  # NegBinomial shape of an empty partition's depth; set from the plateau width
        self.s_max = float(s_max)
        self.e_prior_scale = float(e_prior_scale)
        self.d_loc, self.d_scale, self.d_floor = float(d_loc), float(d_scale), 1.0
        self.register_buffer("log_soup", torch.log(soup))
        self.register_buffer("log_chibar", torch.log(chibar))
        self.register_buffer("soup_feat", torch.log1p(soup * n_genes))
        self.register_buffer("log_e_init", torch.log(e_init))
        # Ambient multiplier of cell-containing partitions relative to empty ones, per pool. 1 until
        # api.calibrate_background sets it from the exclusion bound.
        self.register_buffer("log_kappa", torch.zeros(chibar.shape[0]))
        # Optional log multiplier per barcode, set after calibration from each cell's own counts
        # on genes its cluster does not express. None: every cell of a pool shares log_kappa.
        self.cell_log_kappa = None
        # Optional exclusion constraint, set after calibration: silent_mask[c] marks the genes that
        # cluster c does not express and cell_cluster gives each barcode's cluster (the last row of
        # the mask is empty and stands for "no cluster"). The decoder then cannot explain a cell's
        # counts on those genes as native, so they can only be background and the likelihood itself
        # fixes how much background the cell carries.
        self.silent_mask = None
        self.cell_cluster = None
        self.doublet_like = None  # per barcode: carries a second cell's expression (api.calibrate_background)
        self.background_match = None  # per barcode: profile match of flagged cells (api.calibrate_background)
        self.cluster_match = None  # per barcode: share of a flagged cell's counts its cluster explains (api.calibrate_background)
        self.silent_share = None  # per barcode: share of a flagged cell's counts on its cluster's silent genes (api.calibrate_background)
        self.log_phi = None  # [P, G] log of each gene's own background level as a multiple of the fitted profile (api.calibrate_background), or None
        self.splice_prof = None  # per pool: unspliced shares of background and of own expression per gene (api.calibrate_background)
        self.log_e = nn.Parameter(torch.log(e_init).clone())
        s0 = min(max(s_init / self.s_max, 1e-4), 1.0 - 1e-4)  # s = s_max * sigmoid(logit_s)
        self.logit_s = nn.Parameter(torch.full((chibar.shape[0],), math.log(s0 / (1.0 - s0))))
        self.log_conc = nn.Parameter(torch.tensor(math.log(5e4)))  # dirmult only
        self.soup_enc = nn.Linear(n_genes, n_cond)
        self.encoder = _Encoder(n_genes, n_hidden, n_latent, n_cond)
        self.decoder = _Decoder(n_genes, n_hidden, n_latent, n_cond, torch.log(chibar).mean(dim=0))

    # ---- global parameters -------------------------------------------------------
    def e(self):
        return torch.exp(self.log_e)

    def lk(self, b):
        """Per-barcode log multipliers for a batch of barcode indices, or None when not set."""
        return None if self.cell_log_kappa is None else self.cell_log_kappa[b]

    def sc(self, b):
        """Cluster index of each barcode of a batch for the exclusion constraint, or None when not set."""
        return None if self.cell_cluster is None else self.cell_cluster[b]

    def _native(self, z, cond, cluster):
        log_chi = self.decoder(z, cond)
        if cluster is None or self.silent_mask is None:
            return log_chi
        return torch.log_softmax(log_chi - 30.0 * self.silent_mask[cluster], dim=1)

    def s(self):
        return self.s_max * torch.sigmoid(self.logit_s)

    def global_logprior(self):
        """Prior on the ambient level only; the parsimony prior on s is added in api.fit,
        where the objective is already in nats per UMI."""
        return (-0.5 * ((self.log_e - self.log_e_init) / self.e_prior_scale) ** 2).sum()

    def global_parameters(self):
        return [self.log_e, self.logit_s, self.log_conc]

    # ---- shared forward pieces ---------------------------------------------------
    def _encode(self, x, nz, unit, pool, prior_logit, log_kappa=None):
        r, c, v, _ = nz
        N = x.sum(dim=1).clamp_min(1.0)
        ls = self.log_soup[unit[r], c]
        lc = self.log_chibar[pool[r], c]
        zeros = torch.zeros_like(N)
        llr = zeros.index_add(0, r, v * (lc - ls)) / N
        feats = torch.stack([(torch.log(N) - self.d_loc) / self.d_scale, llr], dim=-1)
        cond = torch.tanh(self.soup_enc(self.soup_feat))[unit]
        z_mu, z_lv, eps_m, eps_ls, y_res = self.encoder(x, feats, cond)
        lkap = self.log_kappa[pool] if log_kappa is None else log_kappa
        eps_m = eps_m + lkap  # the encoder predicts the multiplier relative to kappa
        return N, ls, lc, cond, z_mu, z_lv, eps_m, eps_ls, prior_logit + y_res, lkap

    def _cell_mixture(self, N, e, s, eps, log_chi, nz, ls, lc, pool=None):
        """Log composition probability of every nonzero entry under y = 1, split by source."""
        r, c, _, _ = nz
        if self.log_phi is not None and pool is not None:
            # the gene's own background level inside cells, as a multiple of the fitted profile (exclusion: shared level)
            lphi = self.log_phi[pool[r], c]
            ls, lc = ls + lphi, lc + lphi
        amb = eps * e
        d = (N * (1.0 - s) - amb).clamp_min(self.d_floor)
        swp = s * N
        tot = d + amb + swp
        lp_nat = log_chi[r, c] + torch.log(d / tot)[r]
        lp_amb = ls + torch.log(amb / tot)[r]
        lp_swp = lc + torch.log(swp / tot)[r]
        lp = torch.logsumexp(torch.stack([lp_nat, lp_amb, lp_swp]), dim=0)
        return d, lp, lp_nat, lp_amb, lp_swp

    def _composition(self, v, lp, r, N):
        """Sum over a barcode's nonzero entries of the composition log-likelihood.

        The multinomial coefficient is the same under y = 0 and y = 1 and does not
        depend on any parameter, so it is dropped from both.
        """
        zeros = torch.zeros_like(N)
        if self.likelihood == "multinomial":
            return zeros.index_add(0, r, v * lp)
        conc = torch.exp(self.log_conc)
        ap = conc * torch.exp(lp)
        per = torch.lgamma(v + ap) - torch.lgamma(ap)
        return zeros.index_add(0, r, per) + torch.lgamma(conc) - torch.lgamma(N + conc)

    # ---- training objective ------------------------------------------------------
    def forward(self, x, nz, unit, pool, prior_logit, kl_weight: float = 1.0, sample: bool = True, log_kappa=None, cluster=None):
        r, c, v, _ = nz
        N, ls, lc, cond, z_mu, z_lv, eps_m, eps_ls, y_logit, lkap = self._encode(x, nz, unit, pool, prior_logit, log_kappa)
        eps_sd = torch.exp(eps_ls)
        if sample:
            z = z_mu + torch.exp(0.5 * z_lv) * torch.randn_like(z_mu)
            eps = torch.exp(eps_m + eps_sd * torch.randn_like(eps_m))
        else:
            z, eps = z_mu, torch.exp(eps_m + 0.5 * eps_sd**2)
        e, s = self.e()[unit], self.s()[pool]
        zeros = torch.zeros_like(N)

        # y = 1
        log_chi = self._native(z, cond, cluster)
        d, lp1, _, _, _ = self._cell_mixture(N, e, s, eps, log_chi, nz, ls, lc, pool)
        comp1 = self._composition(v, lp1, r, N)
        logd = torch.log(d)
        size1 = -logd - math.log(self.d_scale) - 0.5 * LOG2PI - 0.5 * ((logd - self.d_loc) / self.d_scale) ** 2
        size1 = size1 + torch.log1p(-s)

        # y = 0
        lp0 = torch.logaddexp(ls + torch.log1p(-s)[r], lc + torch.log(s)[r])
        comp0 = zeros.index_add(0, r, v * lp0)
        a0 = self.empty_alpha
        mean0 = e / (1.0 - s)
        size0 = (
            torch.lgamma(N + a0) - math.lgamma(a0) - torch.lgamma(N + 1.0)
            + a0 * (math.log(a0) - torch.log(a0 + mean0))
            + N * (torch.log(mean0) - torch.log(a0 + mean0))
        )
        a = self.eps_alpha
        log_rate = math.log(a) - lkap  # Gamma(a, a / kappa): prior mean kappa

        # KL terms
        kl_z = 0.5 * (z_mu**2 + torch.exp(z_lv) - 1.0 - z_lv).sum(dim=1)
        e_logp = a * log_rate - math.lgamma(a) + (a - 1.0) * eps_m - torch.exp(log_rate + eps_m + 0.5 * eps_sd**2)
        entropy = eps_m + 0.5 + eps_ls + 0.5 * LOG2PI
        kl_eps = -entropy - e_logp
        q = torch.sigmoid(y_logit)
        kl_y = q * (F.logsigmoid(y_logit) - F.logsigmoid(prior_logit)) + (1.0 - q) * (
            F.logsigmoid(-y_logit) - F.logsigmoid(-prior_logit)
        )

        elbo = q * (comp1 + size1 - kl_weight * kl_z - kl_eps) + (1.0 - q) * (comp0 + size0) - kl_y
        return SimpleNamespace(elbo=elbo, N=N, q=q)

    # ---- posterior summaries for the output stage ---------------------------------
    @torch.no_grad()
    def posterior(self, x, nz, unit, pool, prior_logit, log_kappa=None, cluster=None):
        N, ls, lc, cond, z_mu, _, eps_m, eps_ls, y_logit, _ = self._encode(x, nz, unit, pool, prior_logit, log_kappa)
        eps = torch.exp(eps_m + 0.5 * torch.exp(eps_ls) ** 2)
        e, s = self.e()[unit], self.s()[pool]
        log_chi = self._native(z_mu, cond, cluster)
        _, lp, _, lp_amb, lp_swp = self._cell_mixture(N, e, s, eps, log_chi, nz, ls, lc, pool)
        share_amb = torch.exp(lp_amb - lp)
        share_swp = torch.exp(lp_swp - lp)
        return SimpleNamespace(
            cell_prob=torch.sigmoid(y_logit), eps=eps, z=z_mu, N=N, share_amb=share_amb, share_swp=share_swp
        )
