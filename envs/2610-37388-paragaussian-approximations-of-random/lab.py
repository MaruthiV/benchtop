import numpy as np
import torch
from scipy.special import ndtr

# Constants
N_SAMPLES = 4000
EDGES = np.linspace(-8.0, 8.0, 161)  # bin width 0.1
PHI_C = {"linear": 2.0, "relu": 5.0}  # per-layer growth: Var(Sigma) recursion factor (1 + c/n)


def _cdf_gauss_mix(edges, sigmas):
    # exact conditional-Gaussian mixture CDF: mean_i Phi(x / sigma_i)
    return ndtr(edges[:, None] / sigmas[None, :]).mean(axis=1)


def _masses_from_cdf(cdf):
    return np.concatenate([[cdf[0]], np.diff(cdf), [1.0 - cdf[-1]]])


def _simulate_sigma(n, L, act, seed):
    """Return Sigma^L samples for N random networks (unit-variance input layer z^0 ~ N(0,1)^n).
    Conditionally on layer L-1, the layer-L neuron is N(0, Sigma^L) (paper Remark 1.10)."""
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn(N_SAMPLES, n, 1, generator=g)

    def phi(x):
        return x if act == "linear" else (2.0 ** 0.5) * torch.clamp(x, min=0.0)

    for _ in range(L - 1):
        W = torch.randn(N_SAMPLES, n, n, generator=g)
        z = torch.bmm(W, phi(z)) / (n ** 0.5)
    h = phi(z).squeeze(-1)
    sigma = (h * h).mean(dim=1).double().numpy()
    return sigma


def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    method = config.get("method", "gaussian")
    n = int(config.get("width", 8))
    L = int(config.get("depth", 4))
    act = str(config.get("activation", "relu"))
    if act not in PHI_C:
        act = "relu"

    # Ground truth (identical across methods for a given seed)
    sigma = _simulate_sigma(n, L, act, seed)
    sig_sd = np.sqrt(np.maximum(sigma, 1e-12))
    cdf_mc = _cdf_gauss_mix(EDGES, sig_sd)
    p = _masses_from_cdf(cdf_mc)
    mc_var = float(sigma.mean())
    mc_kurt = float(3.0 * (sigma ** 2).mean() / (sigma.mean() ** 2) - 3.0)

    # Predictors: variance propagated is exactly 1 for normalized phi
    kappa = 3.0 * ((1.0 + PHI_C[act] / n) ** L - 1.0)
    x = EDGES
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    cdf_g = ndtr(x)
    cdf_para = cdf_g - (kappa / 24.0) * (x ** 3 - 3 * x) * pdf  # signed measure CDF

    if method == "gaussian":
        cdf_q = cdf_g
        q = _masses_from_cdf(cdf_q)
        pred_kurt = 0.0
    elif method == "para_gaussian":
        cdf_q = cdf_para
        q = _masses_from_cdf(cdf_q)
        pred_kurt = kappa
    elif method == "para_clipped":
        q = _masses_from_cdf(cdf_para)
        q = np.clip(q, 0.0, None)
        q = q / q.sum()
        cdf_q = np.cumsum(q)[:-1]
        pred_kurt = kappa
    else:
        raise ValueError(f"unknown method {method}")

    tv = 0.5 * float(np.abs(p - q).sum())
    w1 = float(np.abs(cdf_q - cdf_mc).sum() * 0.1)
    return {
        "tv_binned": tv,
        "w1_cdf": w1,
        "mc_excess_kurtosis": mc_kurt,
        "pred_excess_kurtosis": float(pred_kurt),
        "mc_variance": mc_var,
    }
