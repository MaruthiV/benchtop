import math
import functools
import itertools

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

# ---------------------------------------------------------------------------
# Constants: 3 segments, weights on {0.1,...,1.0} (10 levels), anchor constraint
# (at least one weight == 1.0), step limit (each weight moves by <= 1 increment).
# ---------------------------------------------------------------------------
T = 52
D = 3
NL = 10
G = np.linspace(0.1, 1.0, 91)
DG = G[1] - G[0]
T_SHIFT = 10
SIG_SHIFT = 0.1
ALPHA_ADV = 0.2
POOL = 60
TUNE_SEEDS = (10001, 10002, 10003, 10004)
EPS_GRID = (0.1, 0.25, 0.5, 1.0)
BETA_GRID = (0.03, 0.1, 0.3, 1.0)

ARMS = np.array([a for a in itertools.product(range(NL), repeat=D) if max(a) == NL - 1], dtype=int)
N = len(ARMS)
LAM = 0.1 * (ARMS + 1)
SHARE = LAM / LAM.sum(1, keepdims=True)
AR3 = np.arange(D)[None, :]
NB = np.abs(ARMS[:, None, :] - ARMS[None, :, :]).max(2) <= 1
START = int(np.flatnonzero((ARMS == NL - 1).all(1))[0])

# Kernels over arms (fixed hyperparameters, as in the paper)
_dist = np.sqrt(((LAM[:, None, :] - LAM[None, :, :]) ** 2).sum(2))
_ELL = 0.5
_AMP2 = 0.25 ** 2
_s3 = math.sqrt(3.0) * _dist / _ELL
K_MAT = _AMP2 * (1 + _s3) * np.exp(-_s3)
K_RBF = _AMP2 * np.exp(-_dist ** 2 / (2 * _ELL ** 2))
_XC = LAM - 0.55
K_LIN = 0.5 ** 2 * (_XC @ _XC.T)
KERNELS = {
    "gp_ucb_matern_linear": K_MAT + K_LIN,
    "gp_ucb_rbf": K_RBF,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _make_params(rng):
    out = []
    for _ in range(D):
        J = int(rng.integers(5, 10))
        out.append({
            "x": rng.uniform(0.1, 1.0, J),
            "s": rng.uniform(-0.5, 1.0, J),
            "w": rng.uniform(0.05, 0.25, J),
        })
    return out


def _landscape(params):
    R = np.empty((D, NL))
    for i, p in enumerate(params):
        d = -(p["s"][:, None] * np.exp(-(G[None, :] - p["x"][:, None]) ** 2
                                       / (2 * p["w"][:, None] ** 2))).sum(0)
        r = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) * 0.5 * DG)])
        span = r.max() - r.min()
        r = 0.2 + 0.8 * (r - r.min()) / span if span > 1e-9 else np.full_like(r, 0.6)
        R[i] = r[::10]
    return (SHARE * R[AR3, ARMS]).sum(1)


def _build_env(env, rng):
    params = _make_params(rng)
    fs = [_landscape(params)]
    adv = None
    if env == "drift":
        n_ep = (T - 1) // T_SHIFT + 1
        for _ in range(1, n_ep):
            for p in params:
                J = len(p["x"])
                p["x"] = p["x"] + SIG_SHIFT * rng.standard_normal(J)
                p["s"] = p["s"] + SIG_SHIFT * rng.standard_normal(J)
                p["w"] = np.maximum(0.03, p["w"] + SIG_SHIFT * rng.standard_normal(J))
            fs.append(_landscape(params))
    elif env == "reactive":
        F = np.empty((POOL, N))
        for k in range(POOL):
            F[k] = _landscape(_make_params(rng))
        adv = (F, LAM[np.argmin(F, axis=1)])
    return {"f": fs, "adv": adv, "env": env}


def _f_at(e, t, cur):
    if e["env"] == "drift":
        return e["f"][min(t // T_SHIFT, len(e["f"]) - 1)]
    f = e["f"][0]
    if e["env"] == "reactive":
        F, dips = e["adv"]
        c = int(np.argmin(((dips - LAM[cur]) ** 2).sum(1)))
        return (1 - ALPHA_ADV) * f + ALPHA_ADV * F[c]
    return f


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def _move_toward(cur, target):
    nb = np.flatnonzero(NB[cur])
    d = np.abs(ARMS[nb] - ARMS[target]).sum(1)
    return int(nb[int(np.argmin(d))])


def _simulate(kind, hp, env, sigma, delay, dseed, aseed):
    drng = np.random.default_rng(dseed)
    arng = np.random.default_rng(aseed)
    e = _build_env(env, drng)
    noise = sigma * drng.standard_normal(T)
    cur = START
    oa, oy = [], []
    regs = np.zeros(T)
    arms = []
    K = KERNELS.get(kind)
    s2 = max(sigma, 0.05) ** 2 + 1e-8
    for t in range(T):
        f = _f_at(e, t, cur)
        navail = max(0, t - delay)
        u = arng.random()
        r2 = arng.random()
        nbs = np.flatnonzero(NB[cur])
        if navail == 0:
            a = int(nbs[int(r2 * len(nbs))])
        elif kind == "epsilon_greedy":
            tt = t + 1
            eps = min(1.0, hp * tt ** (-1 / 3) * max(math.log(tt), 0.0) ** (1 / 3))
            if u < eps:
                a = int(nbs[int(r2 * len(nbs))])
            else:
                idx = np.array(oa[:navail])
                ys = np.array(oy[:navail])
                cnt = np.bincount(idx, minlength=N)
                sm = np.bincount(idx, weights=ys, minlength=N)
                m = np.where(cnt > 0, sm / np.maximum(cnt, 1), -np.inf)
                a = _move_toward(cur, int(np.argmax(m)))
        else:
            o = np.array(oa[:navail])
            y = np.array(oy[:navail])
            mu0 = y.mean()
            Ko = K[np.ix_(o, o)] + s2 * np.eye(len(o))
            cf = cho_factor(Ko, lower=True)
            al = cho_solve(cf, y - mu0)
            Ks = K[:, o]
            mean = mu0 + Ks @ al
            V = solve_triangular(cf[0], Ks.T, lower=True)
            var = np.maximum(np.diag(K) - (V ** 2).sum(0), 1e-12)
            tt = t + 1
            gam = 0.5 * (1 + math.log(tt)) ** 2
            beta = hp * (1 + math.sqrt(2 * (gam + 1 + math.log(10.0))))
            a = _move_toward(cur, int(np.argmax(mean + beta * np.sqrt(var))))
        regs[t] = f.max() - f[a]
        oa.append(a)
        oy.append(f[a] + noise[t])
        arms.append(a)
        cur = a
    return regs, arms


@functools.lru_cache(maxsize=None)
def _tune(kind, env, sigma, delay):
    grid = EPS_GRID if kind == "epsilon_greedy" else BETA_GRID
    best, best_v = grid[0], float("inf")
    for hp in grid:
        v = 0.0
        for s in TUNE_SEEDS:
            r, _ = _simulate(kind, hp, env, sigma, delay, s, s + 500)
            v += r.sum()
        if v < best_v - 1e-12:
            best, best_v = hp, v
    return best


def run(config: dict, seed: int) -> dict:
    kind = config.get("method", "gp_ucb_matern_linear")
    env = config.get("environment", "stationary")
    sigma = float(config.get("noise_sd", 0.1))
    delay = int(config.get("delay", 0))
    hp = _tune(kind, env, sigma, delay)
    regs, arms = _simulate(kind, hp, env, sigma, delay, int(seed), int(seed) + 1_000_003)
    q = T // 4
    return {
        "cum_regret": float(regs.sum()),
        "early_regret": float(regs[:q].mean()),
        "late_regret": float(regs[-q:].mean()),
        "distinct_arms_frac": float(len(set(arms)) / T),
        "tuned_hp": float(hp),
    }
