import math
import numpy as np
import torch

D = 32
K_TRUE = 4
H = 8
AMPS = np.array([3.0, 2.5, 2.0, 1.5])
N_MAX = 1000
N_TEST = 200
M_GEN = 256
STEPS = 400
LR = 0.02


def _make_data(seed, noise_sd):
    rng = np.random.default_rng(seed)  # data rng only
    t = np.arange(D)
    W = np.stack([a * math.sqrt(2.0 / D) * np.sin(2 * math.pi * (j + 1) * t / D)
                  for j, a in enumerate(AMPS)], axis=1)
    mu = rng.normal(size=D)
    def draw(n):
        z = rng.normal(size=(n, K_TRUE))
        return z @ W.T + mu + noise_sd * rng.normal(size=(n, D))
    Xte = draw(N_TEST)
    Xtr = draw(N_MAX)
    return Xtr, Xte


def _ppca_noise_var(X, q):
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(Xc.T @ Xc / len(X))[::-1]
    return float(max(ev[q:].mean(), 1e-6))


def run(config, seed):
    torch.set_num_threads(1)
    method = config.get("method", "no_skip")
    alpha = float(config.get("alpha", 0.6))
    n = int(config.get("n_train", 200))
    noise_sd = float(config.get("noise_sd", 0.25))

    Xtr_np, Xte_np = _make_data(seed, noise_sd)
    Xtr_np = Xtr_np[:n]
    s02 = _ppca_noise_var(Xtr_np, H)  # base observation variance (PPCA ML)

    if method == "no_skip":
        a = 0.0
        sr2_factor = 1.0          # sigma_r^2 = sigma0^2 (b=1)
    elif method == "skip_same_var":
        a = alpha
        sr2_factor = (1 - alpha) ** 2   # sigma_r^2 = b^2 sigma0^2 -> same obs variance
    elif method == "skip_resid_var":
        a = alpha
        sr2_factor = 1.0          # sigma_r^2 = sigma0^2 -> obs variance sigma0^2/b^2
    else:
        raise ValueError(method)
    b = 1.0 - a
    sr2 = s02 * sr2_factor
    sx2 = sr2 / b ** 2

    # init: separate rng, identical across arms in observation coordinates
    irng = np.random.default_rng(seed + 10_000)
    W0 = irng.normal(size=(D, H)) * 0.3
    E0 = irng.normal(size=(H, D)) * 0.1
    L0 = irng.normal(size=(H, D)) * 0.01
    tf = lambda x: torch.tensor(x, dtype=torch.float32)
    W = (tf(W0) * b).requires_grad_()
    c = torch.zeros(D, requires_grad=True)
    E = tf(E0).requires_grad_()
    e0 = torch.zeros(H, requires_grad=True)
    L = tf(L0).requires_grad_()
    l0 = torch.full((H,), -1.0, requires_grad=True)
    params = [W, c, E, e0, L, l0]
    opt = torch.optim.Adam(params, lr=LR)

    Xtr = tf(Xtr_np)
    Xte = tf(Xte_np)
    gen = torch.Generator().manual_seed(seed + 20_000)  # encoder noise, same across arms

    for _ in range(STEPS):
        mu_q = Xtr @ E.T + e0
        lv = (Xtr @ L.T + l0).clamp(-8, 4)
        eps = torch.randn(mu_q.shape, generator=gen)
        z = mu_q + torch.exp(0.5 * lv) * eps
        res = Xtr - a * Xtr - (z @ W.T + c)
        rec = 0.5 * (res ** 2).sum(1) / sr2 + 0.5 * D * math.log(sr2)
        kl = 0.5 * (mu_q ** 2 + torch.exp(lv) - 1 - lv).sum(1)
        loss = (rec + kl).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    with torch.no_grad():
        # held-out reconstruction MSE (posterior mean)
        mu_te = Xte @ E.T + e0
        res = Xte - a * Xte - (mu_te @ W.T + c)
        mse = float((res ** 2).mean())

        # generation: X = (f(Z) + eps)/b, shared draws across arms
        grng = torch.Generator().manual_seed(seed + 30_000)
        Z = torch.randn(M_GEN, H, generator=grng)
        N = torch.randn(M_GEN, D, generator=grng)
        S = (Z @ W.T + c + math.sqrt(sr2) * N) / b
        d1 = torch.cdist(Xte, S).mean(1)
        self_term = torch.cdist(S, S).sum() / (M_GEN * (M_GEN - 1))
        es = float((d1.mean() - 0.5 * self_term) / math.sqrt(D))

        # exact NLL of the marginal Gaussian (per coordinate)
        Wg = (W / b).double()
        mean = (c / b).double()
        cov = Wg @ Wg.T + sx2 * torch.eye(D, dtype=torch.float64)
        Lc = torch.linalg.cholesky(cov)
        diff = (Xte.double() - mean).T
        sol = torch.linalg.solve_triangular(Lc, diff, upper=False)
        logdet = 2 * torch.log(torch.diagonal(Lc)).sum()
        nll = 0.5 * ((sol ** 2).sum(0).mean() + logdet + D * math.log(2 * math.pi))
        nll = float(nll / D)

    return {
        "energy_score": es,
        "recon_mse": mse,
        "nll_per_dim": nll,
        "obs_var_ratio": float(sx2 / s02),
    }
