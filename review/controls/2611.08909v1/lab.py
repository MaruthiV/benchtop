import math

import numpy as np

# Constants only. No randomness at module level.
TARGET_ACC = 0.574            # Robbins-Monro target acceptance (same for every arm)
ADAPT_GAIN = 0.7              # step-size adaptation gain (burn-in only)
RHO = 0.6                     # SA step exponent: gamma_n = (n+2)^(-RHO)
REFRESH = 10                  # refresh the online Fisher estimate and its factors every REFRESH steps
ALPHA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)   # same grid for SRMC and GA-SRMC
ACC_FLOOR = 0.05              # pilot candidates with lower acceptance are treated as unstable
PILOT_BURN, PILOT_KEEP = 100, 400
EVAL_BURN, EVAL_KEEP = 300, 2000
EPS0 = 0.5                    # initial MALA step size (adapted during burn-in)
EPS_MIN, EPS_MAX = 1e-3, 1.0  # bounds on the adapted step size; keeps the proposal stable
LOG_EPS_MIN, LOG_EPS_MAX = math.log(EPS_MIN), math.log(EPS_MAX)
W_FLOOR = 1e-12               # eigenvalue floor for the online Fisher estimate


def _target_eigs(d, kappa):
    # Log-spaced eigenvalues of Sigma from 1 to kappa (diagonal target).
    return kappa ** (np.arange(d) / float(d - 1))


def _min_ess(X):
    # Minimum-coordinate ESS. Autocorrelations come from FFT, and each coordinate's
    # sum is truncated at its first negative lag. ESS is clipped to [1, T].
    T = X.shape[0]
    Xc = X - X.mean(axis=0)
    nfft = 1 << int(2 * T - 1).bit_length()
    F = np.fft.rfft(Xc, n=nfft, axis=0)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:T]
    var0 = ac[0]
    ok = var0 > 1e-300
    rho = np.zeros_like(ac)
    rho[:, ok] = ac[:, ok] / var0[ok]
    neg = rho[1:] < 0
    has_neg = neg.any(axis=0)
    k = np.where(has_neg, neg.argmax(axis=0) + 1, T)   # first negative lag per coordinate
    lags = np.arange(1, T)[:, None]
    mask = lags < k[None, :]
    tau = 1.0 + 2.0 * np.sum(rho[1:] * mask, axis=0)
    tau = np.clip(tau, 1.0, float(T))
    ess = T / tau
    ess[~ok] = 1.0
    return float(max(np.min(ess), 1.0))


def _refresh(C, cnt, eye, d):
    """
    Online Fisher estimate S_hat = E[s s^T] with a ridge prior of weight 1.
    Returns (S_hat, M_raw, M_norm, L) or None if the estimate is not finite.
    Factorization uses an eigendecomposition with an eigenvalue floor, so it
    cannot raise LinAlgError for a nearly singular or indefinite matrix.
    """
    S_hat = (C + eye) / (cnt + 1.0)
    S_hat = 0.5 * (S_hat + S_hat.T)
    if not np.all(np.isfinite(S_hat)):
        return None
    w, V = np.linalg.eigh(S_hat)
    w = np.maximum(w, W_FLOOR)
    inv_w = 1.0 / w
    M_raw = (V * inv_w) @ V.T            # S_hat^{-1}
    M_raw = 0.5 * (M_raw + M_raw.T)
    # Trace normalization: scale M so that tr(S_hat @ M) = d.
    tr = float(np.trace(S_hat @ M_raw))
    M_norm = (d / tr) * M_raw
    L = V * np.sqrt(inv_w)               # L @ L.T = M_raw
    return S_hat, M_raw, M_norm, L


def _chain(inv_lam, alpha, geo, precond, x0, xi, u, n_burn, n_keep):
    """
    Adaptive MALA-type chain targeting the tilted surrogate
        pi_theta(x) ∝ pi(x) exp(-alpha * theta^T M s(x)),
    which for this Gaussian target is N(m, Sigma) with m = alpha * M @ theta.
    geo=False: M = I (SRMC). geo=True: M = trace-normalized inverse online Fisher (GA-SRMC).
    precond=True: proposal preconditioner A = inverse online Fisher (FisherMALA-style).
    The Metropolis correction is applied to the tilted surrogate with theta held fixed
    within a step. The score average theta is then updated with the SRMC SA rule on
    the true target score. Noise (xi, u) is supplied by the caller, so arms share draws.
    """
    d = x0.shape[0]
    n_total = n_burn + n_keep
    eye = np.eye(d)
    x = x0.copy()
    theta = np.zeros(d)
    C = np.zeros((d, d))          # running sum of s s^T under the true target
    cnt = 0
    A = eye
    Ainv = eye
    L = eye
    M = eye
    log_eps = math.log(EPS0)
    out = np.empty((n_keep, d))
    acc_sum = 0.0

    for t in range(n_total):
        if t % REFRESH == 0:
            fresh = _refresh(C, cnt, eye, d)
            if fresh is not None:
                S_hat, M_raw, M_norm, L_fac = fresh
                if precond:
                    A = M_raw
                    Ainv = S_hat
                    L = L_fac
                if geo:
                    M = M_norm
            # If the estimate is not finite, keep the previous matrices.

        log_eps = min(max(log_eps, LOG_EPS_MIN), LOG_EPS_MAX)
        eps = math.exp(log_eps)
        e2 = eps * eps
        m = alpha * (M @ theta)   # surrogate mean; equals the tilt's effect on the target

        # MALA proposal on the surrogate, with covariance e2 * A.
        g_x = -inv_lam * (x - m)
        mu_x = x + 0.5 * e2 * (A @ g_x)
        y = mu_x + eps * (L @ xi[t])
        g_y = -inv_lam * (y - m)
        mu_y = y + 0.5 * e2 * (A @ g_y)

        dfw = y - mu_x
        drv = x - mu_y
        log_q_fwd = -0.5 * float(dfw @ (Ainv @ dfw)) / e2
        log_q_rev = -0.5 * float(drv @ (Ainv @ drv)) / e2
        lp_x = -0.5 * float(np.sum(inv_lam * (x - m) ** 2))
        lp_y = -0.5 * float(np.sum(inv_lam * (y - m) ** 2))
        log_r = (lp_y - lp_x) + (log_q_rev - log_q_fwd)

        # Non-finite log-ratios are rejected.
        if np.isfinite(log_r) and (log_r >= 0.0 or u[t] < math.exp(log_r)):
            acc = 1.0
            x = y
        else:
            acc = 0.0

        # Score-average SA update uses the true target score at the new state.
        s = -inv_lam * x
        theta = theta + (t + 2.0) ** (-RHO) * (s - theta)
        C += np.outer(s, s)
        cnt += 1

        if t < n_burn:
            log_eps += ADAPT_GAIN * (t + 1.0) ** (-RHO) * (acc - TARGET_ACC)
        else:
            out[t - n_burn] = x
            acc_sum += acc

    return out, acc_sum / float(n_keep)


def run(config: dict, seed: int) -> dict:
    d = int(config["dimension"])
    kappa = float(config["condition_number"])
    method = str(config["method"])
    base = str(config["base_sampler"])

    if d < 2:
        raise ValueError("dimension must be >= 2")
    if method == "SRMC":
        geo = False
    elif method == "GA-SRMC":
        geo = True
    else:
        raise ValueError("unknown method: %s" % method)
    if base not in ("MALA", "FisherMALA"):
        raise ValueError("unknown base_sampler: %s" % base)
    precond = base == "FisherMALA"

    inv_lam = 1.0 / _target_eigs(d, kappa)

    seed = int(seed)
    # Separate streams: rng_init for the start point, rng_noise for all chain noise.
    # Noise and start are independent of the method, so arms share draws (CRN).
    rng_init = np.random.default_rng([seed, 1])
    rng_noise = np.random.default_rng([seed, 2])
    x0 = rng_init.standard_normal(d)

    n_eval = EVAL_BURN + EVAL_KEEP
    n_pilot = PILOT_BURN + PILOT_KEEP
    xi_e = rng_noise.standard_normal((n_eval, d))
    u_e = rng_noise.uniform(size=n_eval)
    xi_p = rng_noise.standard_normal((n_pilot, d))
    u_p = rng_noise.uniform(size=n_pilot)

    # Tune alpha on pilot chains, which use a noise pool separate from evaluation.
    best_alpha, best_score = 0.0, -np.inf
    for alpha in ALPHA_GRID:
        samp, acc = _chain(inv_lam, alpha, geo, precond, x0, xi_p, u_p,
                           PILOT_BURN, PILOT_KEEP)
        if (not np.all(np.isfinite(samp))) or acc < ACC_FLOOR:
            score = -np.inf
        else:
            score = _min_ess(samp) / PILOT_KEEP
        if score > best_score:
            best_alpha, best_score = float(alpha), score

    # Evaluate on fresh samples from the held-out noise pool.
    samp, acc = _chain(inv_lam, best_alpha, geo, precond, x0, xi_e, u_e,
                       EVAL_BURN, EVAL_KEEP)
    samp = np.clip(np.nan_to_num(samp, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    acc = float(acc) if np.isfinite(acc) else 0.0

    ess = _min_ess(samp)
    per_iter = ess / float(EVAL_KEEP)
    # Posterior-mean squared error against the true mean (zero), averaged over coordinates.
    mse = float(np.mean(np.mean(samp, axis=0) ** 2))

    return {
        "log10_min_ess_per_iter": float(math.log10(per_iter)),
        "min_ess_per_iter": float(per_iter),
        "mean_sq_err": mse,
        "accept_rate": acc,
        "alpha": float(best_alpha),
    }

_original_run = run
_original_chain = _chain
_probe_last = {}

def _chain(*args, **kwargs):
    result = _original_chain(*args, **kwargs)
    inv_lam = args[0]
    sample, acceptance = result
    ratios = np.std(sample, axis=0, ddof=1) * np.sqrt(inv_lam)
    _probe_last.update(max_sigma_ratio=float(np.max(ratios)), min_sigma_ratio=float(np.min(ratios)), finite_samples=float(np.all(np.isfinite(sample))), outside_clip_count=float(np.sum(np.abs(sample)>1e6)))
    return result

def run(config, seed):
    if config.get('probe') == 'alpha_zero':
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((80,2))
        u = rng.uniform(size=80)
        args = (np.array([1.0,0.1]), 0.0)
        left = _original_chain(*args,False,bool(config['preconditioned']),np.array([1.0,-1.0]),xi,u,20,60)
        right = _original_chain(*args,True,bool(config['preconditioned']),np.array([1.0,-1.0]),xi,u,20,60)
        return {'max_sample_difference':float(np.max(np.abs(left[0]-right[0]))),'acceptance_difference':float(abs(left[1]-right[1]))}
    if config.get('probe') == 'target_ratio':
        inv_lam=np.array([1.0,0.1]);M=np.array([[2.0,0.3],[0.3,1.0]]);theta=np.array([0.2,-0.4]);alpha=0.3
        x=np.array([0.7,-0.1]);y=np.array([-0.5,0.9]);m=alpha*(M@theta)
        shifted=-0.5*float(np.sum(inv_lam*(y-m)**2)-np.sum(inv_lam*(x-m)**2))
        paper=-0.5*float(np.sum(inv_lam*y*y)-np.sum(inv_lam*x*x))-alpha*float(theta@M@(-inv_lam*y+inv_lam*x))
        return {'ratio_difference':float(abs(shifted-paper))}
    if config.get('probe') == 'ess_scale':
        rng=np.random.default_rng(seed);sample=rng.standard_normal((2000,2))
        return {'ess_original':_min_ess(sample),'ess_rescaled':_min_ess(10.0*sample),'rescaled_max_sigma_ratio':float(np.max(np.std(10*sample,axis=0,ddof=1)))}
    result=_original_run(config,seed)
    result.update({'probe_'+k:v for k,v in _probe_last.items()})
    return result
