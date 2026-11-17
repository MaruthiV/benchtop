import math

import numpy as np
import torch
from scipy.special import expit
from sklearn.datasets import load_breast_cancer

D = 20                    # quadratic dimension
N_TRAIN = 64              # training quadratics (policy fit and baseline threshold grid)
N_TEST = 200              # held-out quadratics
N_LOGREG = 12             # logistic-regression row-subsample instances
LOGREG_FRAC = 0.5         # fraction of breast-cancer rows per instance
HORIZON = 40              # unrolled horizon for training and tuning surrogate
TRAIN_STEPS = 80          # Adam steps for the learned restart policy
LR = 0.05
TAU0, TAU1 = 1.0, 0.05    # relaxation temperature, annealed TAU0 -> TAU1 during training
CAP = 5000                # iteration cap (censoring value)
TOL = 1e-6                # relative gradient-norm tolerance
ODC_GRID = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)
TINY = 1e-300

# Module-level data preparation is deterministic.
_BC = load_breast_cancer()
_XRAW = _BC.data.astype(np.float64)
_XSTD = (_XRAW - _XRAW.mean(axis=0)) / (_XRAW.std(axis=0) + 1e-12)
_XSTD = np.hstack([_XSTD, np.ones((_XSTD.shape[0], 1))])  # intercept column
_YBC = _BC.target.astype(np.float64)


# ---------------------------------------------------------------- problems

def _quadratics(rng, n, kappa):
    """Random quadratics 0.5 x^T A x - b^T x with spectrum exactly in [1, kappa]."""
    lam = np.geomspace(1.0, float(kappa), D)
    Q, _ = np.linalg.qr(rng.standard_normal((n, D, D)))
    A = np.einsum("nij,j,nkj->nik", Q, lam, Q)
    b = rng.standard_normal((n, D))
    x0 = rng.standard_normal((n, D))
    return A, b, x0


def _quad_grad(A, b):
    return lambda Y: np.einsum("nij,nj->ni", A, Y) - b


def _logreg_instances(rng, kappa):
    """Row-subsample instances of standardized breast-cancer logistic regression.

    The L2 weight is set so that L / mu is approximately kappa for each instance.
    """
    m = int(round(LOGREG_FRAC * _XSTD.shape[0]))
    idx = np.stack([rng.permutation(_XSTD.shape[0])[:m] for _ in range(N_LOGREG)])
    X = _XSTD[idx]
    Y = _YBC[idx]
    s2 = np.linalg.svd(X, compute_uv=False)[:, 0] ** 2
    Ls = s2 / (4.0 * m)
    lam = Ls / (float(kappa) - 1.0)
    return X, Y, lam, Ls + lam


def _logreg_grad(X, Y, lam):
    m = X.shape[1]

    def grad(W):
        r = expit(np.einsum("nmp,np->nm", X, W)) - Y
        return np.einsum("nmp,nm->np", X, r) / m + lam[:, None] * W

    return grad


# ---------------------------------------------------------------- rules

def _hard(w):
    """Hard ODC-style rule: restart when the normalized gradient-momentum score exceeds w."""
    return lambda s: (s > w).astype(np.float64)


def _soft(w, sharp):
    """Learned relaxed rule evaluated at the final temperature TAU1."""
    return lambda s: expit(sharp * (s - w) / TAU1)


# ---------------------------------------------------------------- numpy simulation

def _simulate(grad, x0, L, rule, sigma, rng, steps, tol=None):
    """Nesterov with FISTA momentum and a restart rule.

    The restart score is s = <G, x_{k+1} - x_k> / (|G| |x_{k+1} - x_k|), with G the
    (possibly noisy) gradient at y_k. p = rule(s) is the restart weight: momentum is
    scaled by (1 - p) and the momentum counter is pulled toward 1 by p.
    Gradient noise is relative: G = g + sigma * |g| * xi / sqrt(d).
    """
    n, d = x0.shape
    g0 = np.linalg.norm(grad(x0), axis=1)
    x = x0.copy()
    y = x0.copy()
    t = np.ones(n)
    hit_at = np.full(n, float(steps))
    hit = np.zeros(n, dtype=bool)
    logr = 0.0
    rest = 0.0
    done = 0
    for k in range(steps):
        Gt = grad(y)
        gt = np.linalg.norm(Gt, axis=1)
        logr += float(np.mean(np.log(gt / g0 + TINY)))
        if tol is not None:
            newly = (~hit) & (gt <= tol * g0)
            hit_at[newly] = k
            hit |= newly
            if hit.all():
                break
        if sigma > 0.0:
            G = Gt + sigma * gt[:, None] * rng.standard_normal((n, d)) / math.sqrt(d)
        else:
            G = Gt
        xn = y - G / L[:, None]
        dx = xn - x
        denom = np.maximum(np.linalg.norm(G, axis=1) * np.linalg.norm(dx, axis=1), TINY)
        s = np.sum(G * dx, axis=1) / denom
        p = rule(s)
        rest += float(np.mean(p))
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (1.0 - p) * (t - 1.0) / tn
        t = (1.0 - p) * tn + p
        y = xn + beta[:, None] * dx
        x = xn
        done = k + 1
    return hit_at, logr / max(done, 1), rest / max(done, 1)


# ---------------------------------------------------------------- torch training

def _unroll(At, bt, x0t, Lt, w, logk, tau):
    """Differentiable unroll of the same recursion as _simulate (noise-free, soft p).

    Surrogate loss: mean over steps and training quadratics of log(|g_k| / |g_0|).
    Minimizing it maximizes the average log-rate of decay, a smooth proxy for
    iterations-to-tolerance.
    """
    n = x0t.shape[0]
    g0n = torch.linalg.norm(torch.bmm(At, x0t[:, :, None])[:, :, 0] - bt, dim=1)
    x = x0t
    y = x0t
    t = torch.ones(n, dtype=torch.float64)
    sharp = torch.exp(logk)
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(HORIZON):
        G = torch.bmm(At, y[:, :, None])[:, :, 0] - bt
        gn = torch.linalg.norm(G, dim=1)
        total = total + torch.log(gn / g0n + 1e-300).mean()
        xn = y - G / Lt[:, None]
        dx = xn - x
        s = (G * dx).sum(1) / (gn * torch.linalg.norm(dx, dim=1) + 1e-300)
        p = torch.sigmoid(sharp * (s - w) / tau)
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        beta = (1.0 - p) * (t - 1.0) / tn
        t = (1.0 - p) * tn + p
        y = xn + beta[:, None] * dx
        x = xn
    return total / HORIZON


def _train_policy(A, b, x0, kappa, rng_init):
    At = torch.tensor(A, dtype=torch.float64)
    bt = torch.tensor(b, dtype=torch.float64)
    x0t = torch.tensor(x0, dtype=torch.float64)
    Lt = torch.full((A.shape[0],), float(kappa), dtype=torch.float64)
    w = torch.tensor(0.01 * float(rng_init.standard_normal()),
                     dtype=torch.float64, requires_grad=True)
    logk = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w, logk], lr=LR)
    for i in range(TRAIN_STEPS):
        tau = TAU0 * (TAU1 / TAU0) ** (i / (TRAIN_STEPS - 1))
        opt.zero_grad()
        loss = _unroll(At, bt, x0t, Lt, w, logk, tau)
        loss.backward()
        opt.step()
    return float(w.detach()), float(torch.exp(logk).detach())


def _tune_odc(A, b, x0, kappa):
    """Grid-tune the handcrafted gradient-restart threshold on the same training quadratics."""
    grad = _quad_grad(A, b)
    L = np.full(A.shape[0], float(kappa))
    best_w, best = 0.0, math.inf
    for w in ODC_GRID:
        _, logr, _ = _simulate(grad, x0, L, _hard(w), 0.0, None, HORIZON)
        if logr < best:
            best, best_w = logr, w
    return best_w


# ---------------------------------------------------------------- entry point

def run(config: dict, seed: int) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    method = str(config["method"])
    kappa = float(config["condition_number"])
    family = str(config["problem_family_transfer"])
    sigma = float(config["gradient_noise"])

    # Separate streams so the method choice never shifts the data.
    rng_train = np.random.default_rng([seed, 1])
    rng_test = np.random.default_rng([seed, 2])
    rng_noise = np.random.default_rng([seed, 3])
    rng_init = np.random.default_rng([seed, 4])

    A_tr, b_tr, x0_tr = _quadratics(rng_train, N_TRAIN, kappa)

    if family.startswith("logistic"):
        X, Y, lam, L = _logreg_instances(rng_test, kappa)
        grad = _logreg_grad(X, Y, lam)
        x0 = np.zeros((X.shape[0], X.shape[2]))
    else:
        A, b, x0 = _quadratics(rng_test, N_TEST, kappa)
        grad = _quad_grad(A, b)
        L = np.full(A.shape[0], kappa)

    if method == "no_restart":
        rule = _hard(math.inf)
        w_out = 0.0
    elif method == "odc_gradient":
        w_out = _tune_odc(A_tr, b_tr, x0_tr, kappa)
        rule = _hard(w_out)
    elif method == "learned_restart":
        w_out, sharp = _train_policy(A_tr, b_tr, x0_tr, kappa, rng_init)
        rule = _soft(w_out, sharp)
    else:
        raise ValueError(f"unknown method: {method}")

    iters, _, rate = _simulate(grad, x0, L, rule, sigma, rng_noise, CAP, tol=TOL)

    return {
        "iters_median": float(np.median(iters)),
        "iters_p90": float(np.percentile(iters, 90)),
        "frac_capped": float(np.mean(iters >= CAP)),
        "restarts_per_100_iters": float(100.0 * rate),
        "policy_threshold": float(w_out),
    }

_original_run = run

def run(config, seed):
    score = float(config['score'])
    p = float(_soft(0.0, 1.0)(np.array([score]))[0])
    h = float(_hard(0.0)(np.array([score]))[0])
    t = 5.0
    tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
    return {
        'soft_weight': p,
        'hard_weight': h,
        'draft_counter_after_weight': (1.0-p)*tn+p,
        'hard_rule_counter': (1.0-h)*tn+h,
        'full_reset_counter': 1.0,
        'no_reset_counter': tn,
    }
