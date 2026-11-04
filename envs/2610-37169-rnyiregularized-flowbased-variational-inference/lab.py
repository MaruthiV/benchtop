import math
import numpy as np
import torch

torch.set_num_threads(1)

PRIOR_S = 2.0
THETA2_TRUE = 0.5
LAMBDA = 1.0
K = 2          # mixture components
R = 8          # random restarts per run (batched)
NMC = 48       # MC samples per component per step
STEPS = 200
LR = 0.05
GRID_N = 140
GRID_L = 5.0

_g = np.linspace(-GRID_L, GRID_L, GRID_N)
_G1, _G2 = np.meshgrid(_g, _g, indexing="ij")
GRID = torch.tensor(np.stack([_G1.ravel(), _G2.ravel()], -1), dtype=torch.float64)
GRID_POS = torch.tensor(_G1.ravel() > 0)


def _phi(theta, y, sigma):
    # -log unnormalized posterior (Tikhonov objective): data misfit + prior
    r1 = (y[0] - theta[..., 0] ** 2) / sigma
    r2 = (y[1] - theta[..., 1]) / sigma
    return 0.5 * (r1 ** 2 + r2 ** 2) + 0.5 * ((theta / PRIOR_S) ** 2).sum(-1)


def _normal_logpdf(z, mu, sig):
    return (-0.5 * ((z - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)).sum(-1)


def _fit(y, sigma, use_renyi, alpha, seed):
    gen = torch.Generator().manual_seed(10_000 + seed)  # model-init / MC rng, independent of data
    mu = (torch.randn(R, K, 2, generator=gen) * 1.5).requires_grad_(True)
    logsig = torch.full((R, K, 2), math.log(0.7)).requires_grad_(True)
    logit = (0.1 * torch.randn(R, K, generator=gen)).requires_grad_(True)
    opt = torch.optim.Adam([mu, logsig, logit], lr=LR)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(STEPS):
        eps = torch.randn(R, K, NMC, 2, generator=gen)  # drawn regardless of method (CRN)
        opt.zero_grad()
        sig = torch.exp(logsig.clamp(-4.0, 1.5))
        logw = torch.log_softmax(logit, -1)
        w = logw.exp()
        z = mu[:, :, None, :] + sig[:, :, None, :] * eps  # R,K,N,2
        comp = _normal_logpdf(z[:, :, :, None, :], mu[:, None, None, :, :], sig[:, None, None, :, :])  # R,K,N,K
        logq = torch.logsumexp(logw[:, None, None, :] + comp, -1)  # R,K,N
        ephi = (w * _phi(z, yt, sigma).mean(-1)).sum(-1)
        ent = -(w * logq.mean(-1)).sum(-1)
        loss = ephi - ent
        if use_renyi:
            logpi = (-0.5 * (z / PRIOR_S) ** 2 - math.log(PRIOR_S) - 0.5 * math.log(2 * math.pi)).sum(-1)
            d = logpi - logq
            t = torch.logsumexp((logw[:, :, None] + alpha * d - math.log(NMC)).reshape(R, -1), -1)
            loss = loss + LAMBDA * t / (alpha - 1.0)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        return mu.detach().double(), torch.exp(logsig.clamp(-4.0, 1.5)).detach().double(), \
            torch.log_softmax(logit, -1).detach().double()


def run(config: dict, seed: int) -> dict:
    method = config.get("method", "kl")
    a = float(config.get("separation", 1.5))
    sigma = float(config.get("noise", 0.6))
    alpha = float(config.get("alpha", 2.0))

    rng = np.random.default_rng(seed)  # data rng
    y = np.array([a ** 2, THETA2_TRUE]) + sigma * rng.standard_normal(2)

    # grid ground-truth posterior
    yt64 = torch.tensor(y, dtype=torch.float64)
    p = torch.softmax(-_phi(GRID, yt64, sigma), 0)
    pos_true = float(p[GRID_POS].sum())
    minority_true = min(pos_true, 1 - pos_true)

    mu, sig, logw = _fit(y, sigma, method == "renyi", alpha, seed)

    collapses, errs, kls, mins = [], [], [], []
    for r in range(R):
        comp = _normal_logpdf(GRID[:, None, :], mu[r][None], sig[r][None])  # G,K
        lq = torch.logsumexp(logw[r][None] + comp, -1)
        q = torch.softmax(lq, 0)
        qpos = float(q[GRID_POS].sum())
        mn = min(qpos, 1 - qpos)
        mins.append(mn)
        collapses.append(1.0 if mn < 0.10 else 0.0)
        errs.append(abs(qpos - pos_true))
        pe, qe = p.clamp_min(1e-12), q.clamp_min(1e-12)
        kl = float((pe * (pe.log() - qe.log())).sum() + (qe * (qe.log() - pe.log())).sum())
        kls.append(kl)

    return {
        "collapse_frac": float(np.mean(collapses)),
        "mode_mass_err": float(np.mean(errs)),
        "sym_kl": float(np.mean(kls)),
        "mean_minority_mass": float(np.mean(mins)),
        "true_minority_mass": float(minority_true),
    }
